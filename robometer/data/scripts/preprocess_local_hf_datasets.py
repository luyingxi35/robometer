#!/usr/bin/env python3
"""
Preprocess local HuggingFace load_from_disk datasets into Robometer cache format.

Key guarantees:
- Sample frames by video frame count (uniform) and save as .npz
- target_progress is sampled with the same indices and strictly aligned to sampled frames
- partial_success is required
- lang_vector is required; if missing/None/invalid and enabled, compute from task text
"""

import datetime
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

# Default to CPU-safe startup to avoid CUDA device visibility issues in mixed environments.
# Set ROBOMETER_PREPROCESS_ALLOW_CUDA=1 to allow CUDA probing in this script.
if os.environ.get("ROBOMETER_PREPROCESS_ALLOW_CUDA", "0") != "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import decord  # type: ignore
import numpy as np
import torch
from datasets import Dataset, Sequence, Value, load_from_disk
from pyrallis import wrap
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from robometer.utils.distributed import rank_0_print
from robometer.utils.embedding_utils import compute_text_embeddings, compute_video_embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class LocalHFPreprocessConfig:
    # Input datasets
    dataset_roots: list[str] = field(default_factory=lambda: [])
    dataset_names: list[str] = field(default_factory=lambda: [])

    # Processing/caching
    max_frames_for_preprocessing: int = 32
    video_frame_sampling: str = "uniform"
    num_threads: int = 8
    num_proc: int = 1
    force_reprocess: bool = False
    cache_dir: str = ""

    # Required field behavior
    require_partial_success: bool = True
    require_target_progress: bool = True
    require_lang_vector: bool = True
    compute_lang_vector_if_missing: bool = True

    # Embeddings
    precompute_embeddings: bool = False
    embeddings_cache_dir: str = "embeddings"
    dinov2_model: str = "facebook/dinov2-base"
    sentence_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    embedding_batch_size: int = 64


class LocalHFDatasetPreprocessor:
    def __init__(self, config: LocalHFPreprocessConfig):
        self.config = config
        self.datasets: dict[str, Dataset] = {}
        self.dataset_indices: dict[str, dict[str, Any]] = {}
        self.device = self._select_device()

        # Always initialize sentence model if we might need to compute lang_vector.
        self.sentence_model = None
        needs_text_model = config.compute_lang_vector_if_missing or config.precompute_embeddings
        if needs_text_model:
            rank_0_print("Initializing sentence model...")
            self.sentence_model = SentenceTransformer(config.sentence_model)
            self.sentence_model = self.sentence_model.to(self.device)
            rank_0_print(f"Sentence model ready on {self.device}")

        self.dinov2_model = None
        self.dinov2_processor = None
        if config.precompute_embeddings:
            rank_0_print("Initializing DINOv2 model...")
            self.dinov2_model = AutoModel.from_pretrained(config.dinov2_model)
            self.dinov2_processor = AutoImageProcessor.from_pretrained(config.dinov2_model, use_fast=True)
            self.dinov2_model.eval()
            self.dinov2_model = self.dinov2_model.to(self.device)
            rank_0_print(f"DINOv2 model ready on {self.device}")

    def _select_device(self) -> torch.device:
        """Choose a safe torch device without crashing on broken CUDA visibility."""
        try:
            if torch.cuda.is_available():
                # Force a cheap CUDA call; if this fails, we fallback to CPU.
                _ = torch.cuda.current_device()
                return torch.device("cuda")
        except Exception as exc:
            rank_0_print(f"CUDA not usable for preprocessing; fallback to CPU. Reason: {exc}")
        return torch.device("cpu")

    def preprocess(self):
        if not self.config.dataset_roots:
            raise ValueError("dataset_roots is empty")
        if not self.config.cache_dir:
            raise ValueError("cache_dir is empty")

        all_items = self._build_dataset_items()
        self._show_preprocessed_datasets(all_items)

        for idx, (dataset_name, dataset_root) in enumerate(all_items, start=1):
            rank_0_print(f"\nProcessing {idx}/{len(all_items)}: {dataset_name} @ {dataset_root}")
            cache_key = dataset_name
            cache_dir = os.path.join(self.config.cache_dir, cache_key.replace("/", "_").replace(":", "_"))

            if os.path.exists(cache_dir) and not self.config.force_reprocess:
                rank_0_print(f"  Cache exists, loading: {cache_dir}")
                self._load_individual_cache(cache_dir, cache_key)
                continue
            if os.path.exists(cache_dir) and self.config.force_reprocess:
                rank_0_print(f"  force_reprocess=True, removing: {cache_dir}")
                shutil.rmtree(cache_dir)

            dataset = self._load_local_dataset(dataset_root)
            rank_0_print(f"  Loaded {len(dataset)} rows")

            processed_dataset, indices = self._process_dataset_threaded(dataset, cache_dir, cache_key, dataset_root)
            self.datasets[cache_key] = processed_dataset
            self.dataset_indices[cache_key] = indices
            self._save_individual_cache(cache_dir, processed_dataset, indices, dataset_name, dataset_root)

        if not self.datasets:
            raise RuntimeError("No datasets were successfully processed")

        rank_0_print("\nPreprocessing complete")
        total = sum(len(ds) for ds in self.datasets.values())
        rank_0_print(f"Total trajectories: {total}")
        for k, ds in self.datasets.items():
            rank_0_print(f"  {k}: {len(ds)}")

    def _build_dataset_items(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        for i, root in enumerate(self.config.dataset_roots):
            if i < len(self.config.dataset_names) and self.config.dataset_names[i]:
                name = self.config.dataset_names[i]
            else:
                task_name = os.path.basename(os.path.dirname(root.rstrip("/")))
                tail = os.path.basename(root.rstrip("/"))
                name = f"local/{task_name}_{tail}"
            items.append((name, root))
        return items

    def _load_local_dataset(self, dataset_root: str) -> Dataset:
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
        ds = load_from_disk(dataset_root)
        if not isinstance(ds, Dataset):
            raise TypeError(f"Expected Dataset from load_from_disk, got: {type(ds)}")
        return ds

    def _sample_indices_uniform(self, total_frames: int) -> list[int]:
        if total_frames <= 0:
            raise ValueError("video has zero frames")
        m = self.config.max_frames_for_preprocessing
        if total_frames <= m:
            return list(range(total_frames))
        return [int(i * total_frames / m) for i in range(m)]

    def _validate_and_get_progress(self, ex: dict[str, Any], sampled_indices: list[int]) -> list[float]:
        if "target_progress" not in ex:
            raise ValueError(f"missing required field: target_progress for id={ex.get('id')}")

        raw = ex["target_progress"]
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            raise ValueError(f"target_progress must be non-empty sequence for id={ex.get('id')}")

        try:
            progress = [float(v) for v in raw]
        except Exception as exc:
            raise ValueError(f"target_progress contains non-numeric values for id={ex.get('id')}") from exc

        max_idx = max(sampled_indices)
        if max_idx >= len(progress):
            raise ValueError(
                f"target_progress too short after frame sampling for id={ex.get('id')}: "
                f"len(target_progress)={len(progress)}, required>{max_idx}"
            )

        sampled_progress = [progress[i] for i in sampled_indices]
        return sampled_progress

    def _validate_partial_success(self, ex: dict[str, Any]) -> float:
        if "partial_success" not in ex:
            raise ValueError(f"missing required field: partial_success for id={ex.get('id')}")
        val = ex["partial_success"]
        if val is None:
            raise ValueError(f"partial_success is None for id={ex.get('id')}")
        try:
            return float(val)
        except Exception as exc:
            raise ValueError(f"partial_success is non-numeric for id={ex.get('id')}") from exc

    def _is_valid_lang_vector(self, val: Any) -> bool:
        if not isinstance(val, (list, tuple)) or len(val) == 0:
            return False
        try:
            _ = [float(x) for x in val]
            return True
        except Exception:
            return False

    def _compute_lang_vector(self, task_text: str) -> list[float]:
        if self.sentence_model is None:
            raise RuntimeError("sentence_model is not initialized")
        emb = compute_text_embeddings(task_text, self.sentence_model, use_autocast=True, show_progress_bar=False)
        return emb.detach().float().cpu().numpy().tolist()

    def _get_lang_vector(self, ex: dict[str, Any]) -> list[float]:
        existing = ex.get("lang_vector")
        if self._is_valid_lang_vector(existing):
            return [float(x) for x in existing]

        if self.config.require_lang_vector and not self.config.compute_lang_vector_if_missing:
            raise ValueError(f"lang_vector missing/invalid and auto-compute disabled for id={ex.get('id')}")

        task_text = ex.get("task")
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError(f"task missing/invalid; cannot compute lang_vector for id={ex.get('id')}")

        return self._compute_lang_vector(task_text)

    def _compute_video_embeddings(self, frames_array: np.ndarray) -> torch.Tensor:
        if self.dinov2_model is None or self.dinov2_processor is None:
            raise RuntimeError("DINOv2 model not initialized")
        return compute_video_embeddings(
            frames_array,
            self.dinov2_model,
            self.dinov2_processor,
            batch_size=self.config.embedding_batch_size,
            use_autocast=True,
        )

    def _save_embeddings(self, video_embeddings: torch.Tensor, text_embedding: torch.Tensor, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(
            {
                "video_embeddings": video_embeddings,
                "text_embedding": text_embedding,
                "video_shape": tuple(video_embeddings.shape),
                "text_shape": tuple(text_embedding.shape),
            },
            out_path,
        )

    def _resolve_video_path(self, video_path: str, dataset_root: str) -> str:
        """Resolve stale absolute paths in stored rows to the current dataset root layout."""
        if os.path.exists(video_path):
            return video_path

        # dataset_root: .../<TaskName>/hf_dataset
        task_root = os.path.dirname(dataset_root.rstrip("/"))
        task_name = os.path.basename(task_root)

        # Fast path for known structure: .../<label>/videos/<file>.mp4
        if "/videos/" in video_path:
            suffix = video_path.split("/videos/", 1)[1]
            for label in ["successful_labeld", "suboptimal_labeled", "failure_labeled"]:
                candidate = os.path.join(task_root, label, "videos", suffix)
                if os.path.exists(candidate):
                    return candidate

        # Generic fallback: reuse tail from task name onward
        marker = f"/{task_name}/"
        if marker in video_path:
            tail = video_path.split(marker, 1)[1]
            candidate = os.path.join(task_root, tail)
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"video file not found: {video_path}")

    def _process_dataset_threaded(self, dataset: Dataset, cache_dir: str, cache_key: str, dataset_root: str):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        required = ["id", "task", "frames_video", "target_progress", "partial_success"]
        missing_cols = [c for c in required if c not in dataset.column_names]
        if missing_cols:
            raise ValueError(f"Dataset {cache_key} missing required columns: {missing_cols}")

        frames_dir = os.path.join(cache_dir, "frames")
        embeddings_dir = os.path.join(cache_dir, self.config.embeddings_cache_dir)
        os.makedirs(frames_dir, exist_ok=True)
        if self.config.precompute_embeddings:
            os.makedirs(embeddings_dir, exist_ok=True)

        def process_one(i: int):
            ex: dict[str, Any] = dataset[i]
            ex_id = ex.get("id", f"row_{i}")
            video_path = ex.get("frames_video")
            if not isinstance(video_path, str) or not video_path:
                raise ValueError(f"frames_video missing/invalid for id={ex_id}")
            video_path = self._resolve_video_path(video_path, dataset_root)

            vr = decord.VideoReader(video_path, num_threads=1)
            total_frames = len(vr)
            sampled_indices = self._sample_indices_uniform(total_frames)
            frames_array = vr.get_batch(sampled_indices).asnumpy()
            del vr

            sampled_progress = self._validate_and_get_progress(ex, sampled_indices)
            if len(sampled_progress) != int(frames_array.shape[0]):
                raise ValueError(
                    f"sampled target_progress length mismatch for id={ex_id}: "
                    f"len(progress)={len(sampled_progress)}, frames={frames_array.shape[0]}"
                )

            partial_success = self._validate_partial_success(ex)
            lang_vector = self._get_lang_vector(ex)

            frames_filename = f"trajectory_{ex_id}.npz"
            frames_path = os.path.join(frames_dir, frames_filename)
            np.savez_compressed(
                frames_path,
                frames=frames_array,
                shape=frames_array.shape,
                num_frames=int(frames_array.shape[0]),
            )

            emb_path = None
            vid_shape = None
            txt_shape = None
            if self.config.precompute_embeddings:
                video_emb = self._compute_video_embeddings(frames_array)
                text_emb = compute_text_embeddings(ex["task"], self.sentence_model, use_autocast=True, show_progress_bar=False)
                emb_filename = f"trajectory_{i:06d}_{ex_id}_embeddings.pt"
                emb_path = os.path.join(embeddings_dir, emb_filename)
                self._save_embeddings(video_emb, text_emb, emb_path)
                vid_shape = tuple(video_emb.shape)
                txt_shape = tuple(text_emb.shape)

            out = dict(ex)
            out["frames"] = frames_path
            out["frames_shape"] = tuple(frames_array.shape)
            out["num_frames"] = int(frames_array.shape[0])
            out["frames_processed"] = True
            out["target_progress"] = sampled_progress
            out["partial_success"] = partial_success
            out["lang_vector"] = lang_vector
            out.pop("frames_video", None)

            if emb_path is not None:
                out["embeddings_path"] = emb_path
                out["video_embedding_shape"] = vid_shape
                out["text_embedding_shape"] = txt_shape

            return i, out

        idxs = list(range(len(dataset)))
        results: dict[int, dict[str, Any]] = {}
        skipped: list[tuple[int, str, str]] = []

        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            future_to_idx = {executor.submit(process_one, i): i for i in idxs}
            for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=f"Processing {cache_key}", unit="traj"):
                i = future_to_idx[fut]
                try:
                    _, out = fut.result()
                    results[i] = out
                except Exception as exc:
                    ex_id = f"row_{i}"
                    try:
                        ex = dataset[i]
                        ex_id = str(ex.get("id", ex_id))
                    except Exception:
                        pass
                    skipped.append((i, ex_id, str(exc)))

        # Keep original order
        rows = [results[i] for i in idxs if i in results]
        if skipped:
            rank_0_print(f"  Skipped {len(skipped)} invalid trajectories while processing {cache_key}")
            for skipped_idx, ex_id, err in skipped[:10]:
                rank_0_print(f"    - row={skipped_idx} id={ex_id}: {err}")
            if len(skipped) > 10:
                rank_0_print(f"    ... and {len(skipped) - 10} more skipped trajectories")
        if not rows:
            raise RuntimeError(f"All trajectories were skipped for dataset {cache_key}")
        processed_dataset = Dataset.from_list(rows)

        # Keep lang_vector type stable for downstream
        if "lang_vector" in processed_dataset.column_names:
            processed_dataset = processed_dataset.cast_column("lang_vector", Sequence(feature=Value("float32")))

        indices = self._build_indices(rows)
        return processed_dataset, indices

    def _build_indices(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        robot_trajectories: list[int] = []
        human_trajectories: list[int] = []
        optimal_by_task: dict[str, list[int]] = {}
        suboptimal_by_task: dict[str, list[int]] = {}
        quality_indices: dict[str, list[int]] = {}
        task_indices: dict[str, list[int]] = {}
        source_indices: dict[str, list[int]] = {}
        partial_success_indices: dict[str, list[int]] = {}

        for i, ex in enumerate(rows):
            if ex.get("is_robot", True):
                robot_trajectories.append(i)
            else:
                human_trajectories.append(i)

            q = str(ex.get("quality_label", "successful"))
            t = str(ex.get("task", "unknown"))
            s = str(ex.get("data_source", "unknown"))

            quality_indices.setdefault(q, []).append(i)
            task_indices.setdefault(t, []).append(i)
            source_indices.setdefault(s, []).append(i)

            ps = ex.get("partial_success")
            if ps is not None:
                partial_success_indices.setdefault(str(ps), []).append(i)

            optimal_by_task.setdefault(t, [])
            suboptimal_by_task.setdefault(t, [])
            if q in ["successful", "optimal"]:
                optimal_by_task[t].append(i)
            elif q in ["suboptimal", "failed", "failure"]:
                suboptimal_by_task[t].append(i)

        return {
            "robot_trajectories": robot_trajectories,
            "human_trajectories": human_trajectories,
            "optimal_by_task": optimal_by_task,
            "suboptimal_by_task": suboptimal_by_task,
            "quality_indices": quality_indices,
            "task_indices": task_indices,
            "source_indices": source_indices,
            "partial_success_indices": partial_success_indices,
        }

    def _save_individual_cache(
        self,
        cache_dir: str,
        processed_dataset: Dataset,
        indices: dict[str, Any],
        dataset_name: str,
        dataset_root: str,
    ):
        os.makedirs(cache_dir, exist_ok=True)

        dataset_cache_dir = os.path.join(cache_dir, "processed_dataset")
        processed_dataset.save_to_disk(dataset_cache_dir)

        with open(os.path.join(cache_dir, "index_mappings.json"), "w") as f:
            json.dump(indices, f, indent=2)

        info = {
            "dataset_name": dataset_name,
            "dataset_root": dataset_root,
            "total_trajectories": len(processed_dataset),
            "cache_timestamp": str(datetime.datetime.now()),
            "max_frames_for_preprocessing": self.config.max_frames_for_preprocessing,
        }
        with open(os.path.join(cache_dir, "dataset_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        rank_0_print(f"Saved cache: {cache_dir}")

    def _load_individual_cache(self, cache_dir: str, cache_key: str):
        dataset_cache_dir = os.path.join(cache_dir, "processed_dataset")
        mappings_file = os.path.join(cache_dir, "index_mappings.json")
        if not os.path.exists(dataset_cache_dir) or not os.path.exists(mappings_file):
            raise FileNotFoundError(f"Incomplete cache at {cache_dir}")

        self.datasets[cache_key] = Dataset.load_from_disk(dataset_cache_dir)
        with open(mappings_file) as f:
            self.dataset_indices[cache_key] = json.load(f)

    def _show_preprocessed_datasets(self, all_items: list[tuple[str, str]]):
        rank_0_print("Checking existing caches...")
        hit = 0
        for name, _root in all_items:
            cache_key = name.replace("/", "_").replace(":", "_")
            cache_dir = os.path.join(self.config.cache_dir, cache_key)
            if os.path.exists(cache_dir):
                hit += 1
                rank_0_print(f"  HIT  {name}")
            else:
                rank_0_print(f"  MISS {name}")
        rank_0_print(f"Cache hit: {hit}/{len(all_items)}")


@wrap()
def main(config: LocalHFPreprocessConfig):
    try:
        import resource  # type: ignore

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target_soft = min(hard, 65535)
        if soft < target_soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
    except Exception:
        pass

    preprocessor = LocalHFDatasetPreprocessor(config)
    preprocessor.preprocess()

    print("\nDone preprocessing local HF datasets")
    print(f"Set ROBOMETER_PROCESSED_DATASETS_PATH={config.cache_dir}")


if __name__ == "__main__":
    main()
