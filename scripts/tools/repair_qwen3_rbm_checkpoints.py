#!/usr/bin/env python3
"""Rebuild malformed Qwen3 RBM checkpoints with the correct config and dtype.

The affected checkpoints contain Qwen3 weights, but an old RBM.config_class
caused Transformers to serialize them as Qwen2.5-VL and FSDP left the saved
weights in float32. This script reconstructs RBM from the configured Qwen3 base
model, strictly loads every checkpoint tensor, casts the model, and saves a
fresh Hugging Face checkpoint.

By default, ``checkpoint-100`` is written as ``checkpoint-100-fixed``. Pass
``--in-place`` to atomically replace each original path while retaining the
original directory as ``checkpoint-100-broken-config-backup``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

from robometer.utils.save import load_model_from_hf


CHECKPOINT_PATHS = [
    Path("/data/yingxi/robometer/outputs/robometer_5task/robometer_5task_exp1/checkpoint-100"),
    Path("/data/yingxi/robometer/outputs/robometer_5task/robometer_5task_exp1/checkpoint-200"),
    Path("/data/yingxi/robometer/outputs/robometer_5task/robometer_5task_exp1/checkpoint-300"),
    Path("/data/yingxi/robometer/outputs/robometer_5task/robometer_5task_exp1/checkpoint-400"),
]

MODEL_ARTIFACT_PREFIXES = ("model-", "pytorch_model-")
MODEL_ARTIFACT_NAMES = {
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        default=CHECKPOINT_PATHS,
        help="Checkpoint directories. Defaults to the four 5-task checkpoints declared in this script.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used after loading (for example cpu or cuda:0). CPU is the safer default for a 4B model.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Floating-point dtype of the repaired checkpoint.",
    )
    parser.add_argument("--max-shard-size", default="5GB", help="Maximum output safetensors shard size.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace each original path after validation and retain a *-broken-config-backup directory.",
    )
    return parser.parse_args()


def _is_model_artifact(name: str) -> bool:
    return name in MODEL_ARTIFACT_NAMES or name.startswith(MODEL_ARTIFACT_PREFIXES)


def _link_or_copy(source: str, destination: str) -> str:
    """Hard-link large trainer state when possible, otherwise copy it."""
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _copy_auxiliary_artifacts(source: Path, destination: Path) -> None:
    """Preserve tokenizer, processor, Trainer, optimizer, and RNG artifacts."""
    destination.mkdir(parents=False, exist_ok=False)
    for child in source.iterdir():
        if _is_model_artifact(child.name):
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, copy_function=_link_or_copy)
        else:
            _link_or_copy(str(child), str(target))


def _find_training_config(checkpoint: Path) -> Path | None:
    current = checkpoint
    for _ in range(7):
        candidate = current / "config.yaml"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def _ensure_processor_artifacts(processor, tokenizer, destination: Path) -> None:
    """Fill processor/tokenizer files absent from the original checkpoint."""
    with tempfile.TemporaryDirectory(prefix=".rbm-processor-", dir=destination.parent) as tmp:
        tmp_path = Path(tmp)
        processor.save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)
        for child in tmp_path.iterdir():
            target = destination / child.name
            if not target.exists():
                shutil.copy2(child, target)


def _checkpoint_tensor_signatures(checkpoint: Path) -> tuple[dict[str, tuple[int, ...]], set[str]]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as file:
            index = json.load(file)
        tensor_files = sorted({checkpoint / name for name in index["weight_map"].values()})
    else:
        tensor_files = sorted(checkpoint.glob("model*.safetensors"))

    if not tensor_files:
        raise RuntimeError(f"No model safetensors found in {checkpoint}")

    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: set[str] = set()
    for tensor_file in tensor_files:
        with safe_open(tensor_file, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                tensor_slice = tensors.get_slice(key)
                shapes[key] = tuple(tensor_slice.get_shape())
                dtypes.add(str(tensor_slice.get_dtype()).lower())
    return shapes, dtypes


def _validate_repaired_checkpoint(source: Path, repaired: Path, expected_dtype: str) -> None:
    with (repaired / "config.json").open() as file:
        config = json.load(file)

    errors = []
    if config.get("model_type") != "qwen3_vl":
        errors.append(f"model_type={config.get('model_type')!r}")
    if config.get("architectures") != ["RBM"]:
        errors.append(f"architectures={config.get('architectures')!r}")
    if config.get("text_config", {}).get("model_type") != "qwen3_vl_text":
        errors.append(f"text model_type={config.get('text_config', {}).get('model_type')!r}")
    if config.get("dtype") != expected_dtype:
        errors.append(f"dtype={config.get('dtype')!r}")
    if "qwen2_5_vl" in json.dumps(config):
        errors.append("config still contains qwen2_5_vl")
    if errors:
        raise RuntimeError(f"Invalid repaired config in {repaired}: {', '.join(errors)}")

    source_shapes, _ = _checkpoint_tensor_signatures(source)
    repaired_shapes, repaired_dtypes = _checkpoint_tensor_signatures(repaired)
    if source_shapes != repaired_shapes:
        missing = sorted(source_shapes.keys() - repaired_shapes.keys())[:10]
        unexpected = sorted(repaired_shapes.keys() - source_shapes.keys())[:10]
        changed = sorted(
            key
            for key in source_shapes.keys() & repaired_shapes.keys()
            if source_shapes[key] != repaired_shapes[key]
        )
        raise RuntimeError(
            "Repaired tensor manifest differs from the source: "
            f"missing={missing}, unexpected={unexpected}, changed_shapes={changed[:10]}"
        )

    expected_safetensors_dtype = {
        "bfloat16": "bf16",
        "float16": "f16",
        "float32": "f32",
    }[expected_dtype]
    if repaired_dtypes != {expected_safetensors_dtype}:
        raise RuntimeError(
            f"Unexpected tensor dtypes in {repaired}: {sorted(repaired_dtypes)}; "
            f"expected only {expected_safetensors_dtype}"
        )

    index_path = repaired / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as file:
            metadata = json.load(file).get("metadata", {})
        expected_parameters = sum(math.prod(shape) for shape in repaired_shapes.values())
        if metadata.get("total_parameters") != expected_parameters:
            raise RuntimeError(
                f"Incorrect total_parameters metadata: {metadata.get('total_parameters')} != {expected_parameters}"
            )


def _publish_repaired_checkpoint(source: Path, staging: Path, in_place: bool) -> Path:
    if not in_place:
        target = source.with_name(f"{source.name}-fixed")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing repaired checkpoint: {target}")
        staging.rename(target)
        return target

    backup = source.with_name(f"{source.name}-broken-config-backup")
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")

    source.rename(backup)
    try:
        staging.rename(source)
    except BaseException:
        backup.rename(source)
        raise
    return source


def repair_checkpoint(checkpoint: Path, *, device: str, dtype: str, max_shard_size: str, in_place: bool) -> Path:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    staging = checkpoint.with_name(f".{checkpoint.name}-repairing")
    if staging.exists():
        raise FileExistsError(f"Remove stale staging directory before retrying: {staging}")
    if in_place:
        backup = checkpoint.with_name(f"{checkpoint.name}-broken-config-backup")
        if backup.exists():
            raise FileExistsError(f"Refusing to overwrite existing backup: {backup}")
    else:
        target = checkpoint.with_name(f"{checkpoint.name}-fixed")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing repaired checkpoint: {target}")

    print(f"[repair] loading {checkpoint}")
    exp_config, tokenizer, processor, model = load_model_from_hf(
        str(checkpoint),
        device=torch.device(device),
        use_unsloth=False,
    )
    if exp_config is None or "qwen3" not in exp_config.model.base_model_id.lower():
        raise RuntimeError(f"Checkpoint does not declare a Qwen3 base model: {checkpoint}")

    target_dtype = getattr(torch, dtype)
    model = model.to(dtype=target_dtype)

    try:
        _copy_auxiliary_artifacts(checkpoint, staging)
        training_config = _find_training_config(checkpoint)
        if training_config is not None and not (staging / "config.yaml").exists():
            shutil.copy2(training_config, staging / "config.yaml")

        model.save_pretrained(
            staging,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        _ensure_processor_artifacts(processor, tokenizer, staging)
        _validate_repaired_checkpoint(checkpoint, staging, dtype)
    except BaseException:
        print(f"[repair] failed; staging directory retained for inspection: {staging}")
        raise
    finally:
        del model, processor, tokenizer, exp_config
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target = _publish_repaired_checkpoint(checkpoint, staging, in_place)
    print(f"[repair] completed: {target}")
    return target


def main() -> None:
    args = _parse_args()
    if not args.checkpoints:
        raise ValueError("At least one checkpoint path is required.")

    repaired = []
    for checkpoint in args.checkpoints:
        repaired.append(
            repair_checkpoint(
                checkpoint,
                device=args.device,
                dtype=args.dtype,
                max_shard_size=args.max_shard_size,
                in_place=args.in_place,
            )
        )

    print("[repair] all checkpoints completed:")
    for checkpoint in repaired:
        print(f"  {checkpoint}")


if __name__ == "__main__":
    main()
