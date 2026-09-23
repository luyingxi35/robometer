#!/usr/bin/env python3
"""Compare two Robometer checkpoints on the StackCube failure test set.

The source labels contain 300 action-aligned progress values and the render
video contains one leading reset frame (301 frames total). This script keeps
that alignment explicit, saves all 32 preprocessed render frames per sample,
and uses the same 8-frame inference subsampling as the five-task eval.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("ROBOMETER_PROCESSED_DATASETS_PATH", "/data/yingxi/robometer")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DEFAULT_ROOT = Path("/data/yingxi/robometer/natural_five_task_20260921")
DEFAULT_HF_DATASET = DEFAULT_ROOT / "local_hf/natural_five_task_test"
DEFAULT_PROCESSED_DATASET = DEFAULT_ROOT / "processed/local_natural_five_task_test/processed_dataset"
DEFAULT_TRAINED = DEFAULT_ROOT / "training/natural_five_task_fsdp_pilot/checkpoint-400"
DEFAULT_BASE = Path("/data/yingxi/robometer/robometer-4b_basefixed")


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2) + "\n")


def sample_indices(length: int, count: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("cannot sample an empty sequence")
    if length <= count:
        return np.arange(length, dtype=int)
    return np.round(np.linspace(0, length - 1, count)).astype(int)


def preprocessing_indices(length: int, count: int) -> np.ndarray:
    """Match preprocess_local_hf_datasets._sample_indices_uniform exactly."""
    if length <= 0:
        raise ValueError("cannot sample an empty sequence")
    if length <= count:
        return np.arange(length, dtype=int)
    return np.asarray([int(i * length / count) for i in range(count)], dtype=int)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return result


def spearman(pred: list[float], target: list[float]) -> float | None:
    x = np.asarray(pred, dtype=float)
    y = np.asarray(target, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def load_rows(hf_path: Path, processed_path: Path, limit: int | None) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    hf = load_from_disk(str(hf_path))
    processed = load_from_disk(str(processed_path))
    processed_by_id = {str(row["id"]): row for row in processed}
    rows = []
    for row in hf:
        row_id = str(row.get("id", ""))
        if not row_id.startswith("StackCube-v1_test_failure_"):
            continue
        if row_id not in processed_by_id:
            raise KeyError(f"missing processed row for {row_id}")
        rows.append({"source": row, "processed": processed_by_id[row_id]})
    rows.sort(key=lambda item: str(item["source"]["id"]))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError("no StackCube failure rows found")
    return rows


def prepare_row(item: dict[str, Any], max_frames: int) -> dict[str, Any]:
    from robometer.data.datasets.helpers import load_frames_from_npz

    source = item["source"]
    processed = item["processed"]
    row_id = str(source["id"])
    video_path = Path(str(source["frames_video"]))
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    frames = load_frames_from_npz(str(processed["frames"]))
    processed_count = len(frames)
    source_progress = np.asarray(source["target_progress"], dtype=float)
    metadata = dict(source.get("metadata") or {})
    offset = int(
        metadata.get(
            "video_frame_offset",
            processed.get("metadata", {}).get("video_frame_offset", 0),
        )
    )
    progress_idx = preprocessing_indices(len(source_progress), processed_count)
    render_idx = progress_idx + offset
    source_frame_count = int(metadata.get("source_frame_count", render_idx[-1] + 1))
    if len(render_idx) != processed_count or render_idx[-1] >= source_frame_count:
        raise ValueError(f"invalid source alignment for {row_id}")
    target_32 = source_progress[progress_idx]
    model_idx = sample_indices(processed_count, max_frames)
    return {
        "id": row_id,
        "task": str(source.get("task", "")),
        "video_path": str(video_path),
        "frames": frames,
        "target_32": target_32,
        "progress_indices_32": progress_idx,
        "render_indices_32": render_idx,
        "model_indices": model_idx,
        "model_frames": frames[model_idx],
        "model_target": target_32[model_idx],
        "video_frame_offset": offset,
    }


def make_sample(row: dict[str, Any]):
    from robometer.data.dataset_types import ProgressSample, Trajectory

    trajectory = Trajectory(
        frames=row["model_frames"],
        frames_shape=tuple(row["model_frames"].shape),
        target_progress=[float(v) for v in row["model_target"]],
        predict_last_frame_mask=[1.0] * len(row["model_frames"]),
        success_label=[0.0] * len(row["model_frames"]),
        task=row["task"],
        id=row["id"],
        quality_label="failure_labeled",
        data_source="local_natural_five_task_test",
    )
    return ProgressSample(trajectory=trajectory, sample_type="progress")


def infer(rows: list[dict[str, Any]], checkpoint: Path, batch_size: int) -> dict[str, list[float]]:
    from robometer.evals.baselines.rbm_model import RBMModel

    print(f"[load] {checkpoint}", flush=True)
    model = RBMModel(str(checkpoint))
    predictions: dict[str, list[float]] = {}
    samples = [make_sample(row) for row in rows]
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        values = model.compute_batched_progress(batch)
        for row, prediction in zip(rows[start : start + batch_size], values):
            predictions[row["id"]] = [float(v) for v in prediction]
        print(
            f"[infer] {checkpoint.name}: "
            f"{min(start + batch_size, len(samples))}/{len(samples)}",
            flush=True,
        )
    del model
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def read_render_frames(video_path: str, indices: np.ndarray) -> list[np.ndarray]:
    import decord

    reader = decord.VideoReader(video_path, num_threads=1)
    return [frame for frame in reader.get_batch(indices.tolist()).asnumpy()]


def save_all_render_frames(row: dict[str, Any], directory: Path) -> list[str]:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    frames = read_render_frames(row["video_path"], row["render_indices_32"])
    paths = []
    for progress_index, render_index, frame in zip(
        row["progress_indices_32"], row["render_indices_32"], frames
    ):
        path = directory / f"frame_{int(progress_index):04d}_render_{int(render_index):04d}.png"
        Image.fromarray(frame).convert("RGB").save(path)
        paths.append(str(path))
    return paths


def plot_dashboard(
    row: dict[str, Any],
    trained: list[float],
    base: list[float],
    output_png: Path,
    output_jpg: Path,
) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    model_positions = row["model_indices"]
    render_indices = row["render_indices_32"][model_positions]
    progress_indices = row["progress_indices_32"][model_positions]
    frames = read_render_frames(row["video_path"], render_indices)
    target = np.asarray(row["model_target"], dtype=float)
    trained = np.asarray(trained, dtype=float)[: len(model_positions)]
    base = np.asarray(base, dtype=float)[: len(model_positions)]
    ncols = len(frames)
    fig = plt.figure(figsize=(max(14.0, ncols * 1.8), 9.5))
    grid = fig.add_gridspec(
        2, ncols, height_ratios=[1.0, 2.5], hspace=0.02, wspace=0.025
    )
    for col, (frame, progress_index, render_index) in enumerate(
        zip(frames, progress_indices, render_indices)
    ):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(Image.fromarray(frame).convert("RGB"))
        ax.set_title(
            f"t={int(render_index)}\nlabel={int(progress_index)}",
            fontsize=7,
            pad=2,
        )
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :])
    x = np.arange(ncols)
    ax.plot(x, target, "o-", color="#222222", linewidth=2.0, markersize=4, label="label")
    ax.plot(x, trained, "s-", color="#d95f02", linewidth=2.0, markersize=4, label="ckpt-400")
    ax.plot(x, base, "^-", color="#1b9e77", linewidth=2.0, markersize=4, label="basefixed")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in render_indices], rotation=45, ha="right")
    ax.set_xlabel("render-camera timestep")
    ax.set_ylabel("progress")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title(row["id"], fontsize=10)
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.18, top=0.92)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    fig.savefig(output_jpg, dpi=180, bbox_inches="tight", pil_kwargs={"quality": 95})
    plt.close(fig)


def evaluate(row: dict[str, Any], prediction: list[float]) -> dict[str, Any]:
    target = np.asarray(row["model_target"], dtype=float)
    pred = np.asarray(prediction, dtype=float)
    n = min(len(target), len(pred))
    target, pred = target[:n], pred[:n]
    return {
        "mae": float(np.mean(np.abs(target - pred))),
        "spearman": spearman(pred.tolist(), target.tolist()),
        "prediction": pred.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dataset", type=Path, default=DEFAULT_HF_DATASET)
    parser.add_argument("--processed-dataset", type=Path, default=DEFAULT_PROCESSED_DATASET)
    parser.add_argument("--trained-checkpoint", type=Path, default=DEFAULT_TRAINED)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = load_rows(args.hf_dataset, args.processed_dataset, args.limit)
    rows = [prepare_row(item, args.max_frames) for item in items]
    for row in rows:
        if len(row["model_frames"]) != args.max_frames:
            raise ValueError(f"unexpected model frame count for {row['id']}")

    trained_predictions = infer(rows, args.trained_checkpoint, args.batch_size)
    base_predictions = infer(rows, args.base_checkpoint, args.batch_size)
    all_records = []
    trained_metrics = []
    base_metrics = []
    for row in rows:
        trajectory_dir = args.output_dir / "per_trajectory" / row["id"]
        frame_paths = save_all_render_frames(row, trajectory_dir / "render_camera_frames")
        trained_result = evaluate(row, trained_predictions[row["id"]])
        base_result = evaluate(row, base_predictions[row["id"]])
        plot_dashboard(
            row,
            trained_result["prediction"],
            base_result["prediction"],
            trajectory_dir / "dashboard.png",
            trajectory_dir / "dashboard.jpg",
        )
        record = {
            "id": row["id"],
            "video_path": row["video_path"],
            "video_frame_offset": row["video_frame_offset"],
            "progress_indices_32": row["progress_indices_32"].tolist(),
            "render_indices_32": row["render_indices_32"].tolist(),
            "model_indices": row["model_indices"].tolist(),
            "model_render_indices": row["render_indices_32"][row["model_indices"]].tolist(),
            "target_progress_32": row["target_32"].tolist(),
            "target_progress_model_points": row["model_target"].tolist(),
            "render_camera_frames": frame_paths,
            "trained": {"checkpoint": str(args.trained_checkpoint), **trained_result},
            "basefixed": {"checkpoint": str(args.base_checkpoint), **base_result},
            "dashboard_png": str(trajectory_dir / "dashboard.png"),
            "dashboard_jpg": str(trajectory_dir / "dashboard.jpg"),
        }
        all_records.append(record)
        trained_metrics.append(trained_result)
        base_metrics.append(base_result)

    def aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
        spears = [v["spearman"] for v in values if v["spearman"] is not None]
        return {
            "n": len(values),
            "mean_mae": float(np.mean([v["mae"] for v in values])),
            "mean_spearman": float(np.mean(spears)) if spears else None,
            "valid_spearman_n": len(spears),
        }

    metrics = {
        "trained_ckpt400": aggregate(trained_metrics),
        "basefixed": aggregate(base_metrics),
    }
    write_json(
        args.output_dir / "run_config.json",
        {
            "hf_dataset": args.hf_dataset,
            "processed_dataset": args.processed_dataset,
            "trained_checkpoint": args.trained_checkpoint,
            "base_checkpoint": args.base_checkpoint,
            "max_frames": args.max_frames,
            "batch_size": args.batch_size,
            "num_rows": len(rows),
            "alignment": (
                "progress_indices = floor(i * 300 / 32); "
                "render_indices = progress_indices + video_frame_offset"
            ),
        },
    )
    write_json(args.output_dir / "records.json", all_records)
    write_json(args.output_dir / "metrics.json", metrics)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "trained_mae",
                "trained_spearman",
                "basefixed_mae",
                "basefixed_spearman",
            ]
        )
        for record in all_records:
            writer.writerow(
                [
                    record["id"],
                    record["trained"]["mae"],
                    record["trained"]["spearman"],
                    record["basefixed"]["mae"],
                    record["basefixed"]["spearman"],
                ]
            )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
