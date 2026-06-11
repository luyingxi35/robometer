#!/usr/bin/env python3
"""Last-frame MAE evaluation for RBM reward models.

Computes "末帧准确率" — MAE between the model's predicted progress at the
final (last) frame and the ground-truth progress value at that frame.

Two settings evaluated in a single run:

  Setting 1 (full video):
      Input the full trajectory; compare pred[-1] vs gt[-1].

  Setting 2 (truncated video, 5 seeds):
      For each trajectory, randomly sample 5 cutoff points k ∈ [N//4, N//2].
      Input only frames [0:k]; compare pred[-1] vs gt[k-1].
      Final metric = mean over (trajectories × seeds).

Usage:
    cd ~/RoboFAC/robometer
    uv run python evals/run_last_frame_mae_eval.py \\
        --model-path /data/yingxi/robometer/logs/rbm/checkpoint-100/ \\
        --output-dir /data/yingxi/robometer/all_checkpoint_last_frame_mae_output/checkpoint-100
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ── env setup (mirror run_baseline_eval.py) ──────────────────────────────────
# Allow caller to set these; fall back to defaults used in sweep scripts.
os.environ.setdefault("ROBOMETER_PROCESSED_DATASETS_PATH", "/data/yingxi/robometer")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SUCCESSFUL_LABELS = {"successful", "successful_labeled"}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True,
                   help="Path to RBM checkpoint (local dir or HF repo ID).")
    p.add_argument("--output-dir", default=None,
                   help="Where to write results. Defaults to "
                        "./last_frame_mae_output/<model_slug>/")
    p.add_argument("--dataset-name", default="local_PegInsertionVertical_eval",
                   help="Processed dataset name under ROBOMETER_PROCESSED_DATASETS_PATH.")
    p.add_argument("--max-trajectories", type=int, default=None,
                   help="Limit number of trajectories (useful for quick smoke tests).")
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Number of random truncation seeds per trajectory (Setting 2).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Inference batch size.")
    p.add_argument("--random-seed", type=int, default=42,
                   help="Base random seed for reproducible truncation sampling.")
    p.add_argument("--max-frames", type=int, default=8,
                   help="Max frames passed to the model (subsample longer videos to this length).")
    p.add_argument("--quality-labels", nargs="+",
                   default=["successful", "successful_labeled"],
                   help="Quality labels to evaluate (space-separated). "
                        "E.g. --quality-labels failure_labeled suboptimal_labeled")
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _slug(model_path: str) -> str:
    parts = model_path.rstrip("/").split("/")
    slug = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return re.sub(r"[^\w\-]", "_", slug)


def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    return obj


def _run_batched(
    model,
    samples: List[Any],
    batch_size: int,
    desc: str = "Inference",
) -> List[List[float]]:
    """Run model.compute_batched_progress in mini-batches; return flat list of preds."""
    all_preds: List[List[float]] = []
    for start in tqdm(range(0, len(samples), batch_size), desc=desc):
        batch = samples[start : start + batch_size]
        all_preds.extend(model.compute_batched_progress(batch))
    return all_preds


# ── dataset loading ───────────────────────────────────────────────────────────

def _load_rows(dataset_name: str, quality_labels: list, max_trajectories: Optional[int]):
    """Load the processed HuggingFace dataset and filter by quality_labels."""
    from datasets import load_from_disk

    cache_dir = os.environ.get("ROBOMETER_PROCESSED_DATASETS_PATH", "")
    if not cache_dir:
        raise ValueError("ROBOMETER_PROCESSED_DATASETS_PATH is not set.")

    dataset_path = os.path.join(cache_dir, dataset_name, "processed_dataset")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_from_disk(dataset_path)
    quality_set = set(quality_labels)
    rows = [row for row in dataset if row.get("quality_label") in quality_set]

    if max_trajectories is not None:
        rows = rows[:max_trajectories]

    labels_str = ", ".join(sorted(quality_set))
    print(f"Loaded {len(rows)} trajectories ({labels_str}) from '{dataset_name}'")
    return rows


# ── sample construction ───────────────────────────────────────────────────────

def _subsample_frames(frames: np.ndarray, max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample frames to at most max_frames, always keeping the last frame.

    Returns (subsampled_frames, indices_into_original).
    """
    n = len(frames)
    if n <= max_frames:
        return frames, np.arange(n)
    # np.linspace includes both endpoints, so index[-1] == n-1 (last original frame)
    indices = np.round(np.linspace(0, n - 1, max_frames)).astype(int)
    return frames[indices], indices


def _make_progress_sample(frames: np.ndarray, row: Dict[str, Any], max_frames: int = 8):
    """Create a ProgressSample from raw frames and a dataset row.

    Subsamples frames to max_frames so every sample in a batch has the same
    number of visual tokens (required by the VLM collator).
    """
    from robometer.data.dataset_types import ProgressSample, Trajectory

    subsampled, indices = _subsample_frames(frames, max_frames)
    n_sub = len(subsampled)

    target_progress_full = list(row.get("target_progress") or [])
    # Map each selected frame index to its GT progress value
    target_progress_sub = [
        float(target_progress_full[i]) if i < len(target_progress_full) else 0.0
        for i in indices
    ]

    traj = Trajectory(
        frames=subsampled,
        frames_shape=tuple(subsampled.shape),       # required by collator
        target_progress=target_progress_sub,
        predict_last_frame_mask=[1.0] * n_sub,      # all frames eligible
        success_label=[0.0] * n_sub,                # dummy — not used in inference
        task=row.get("task", ""),
        id=str(row.get("id", "")),
        quality_label=row.get("quality_label", ""),
        data_source=row.get("data_source", ""),
        partial_success=row.get("partial_success"),
    )
    return ProgressSample(trajectory=traj, sample_type="progress")


# ── main eval logic ───────────────────────────────────────────────────────────

def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    from robometer.data.datasets.helpers import load_frames_from_npz
    from robometer.evals.baselines.rbm_model import RBMModel

    # ── model ──
    print(f"\nLoading model from: {args.model_path}")
    model = RBMModel(checkpoint_path=args.model_path)

    max_frames = args.max_frames
    print(f"max_frames: {max_frames}")

    # ── data ──
    rows = _load_rows(args.dataset_name, args.quality_labels, args.max_trajectories)
    if not rows:
        print("No successful trajectories found — aborting.", file=sys.stderr)
        sys.exit(1)

    # ── pre-load all frames ──────────────────────────────────────────────────
    print("Loading frames …")
    all_frames: List[np.ndarray] = []
    valid_rows: List[Dict] = []
    for row in tqdm(rows, desc="Load frames"):
        frames_path = row.get("frames")
        if not isinstance(frames_path, str) or not os.path.exists(frames_path):
            print(f"  [skip] id={row.get('id')}: frames path missing or invalid ({frames_path})",
                  file=sys.stderr)
            continue
        try:
            frames = load_frames_from_npz(frames_path)  # [N, H, W, C] uint8
        except Exception as e:
            print(f"  [skip] id={row.get('id')}: {e}", file=sys.stderr)
            continue
        if len(frames) < 4:
            print(f"  [skip] id={row.get('id')}: too few frames ({len(frames)})", file=sys.stderr)
            continue
        all_frames.append(frames)
        valid_rows.append(row)

    print(f"{len(valid_rows)} trajectories loaded successfully.")

    # ── Setting 1: full video ────────────────────────────────────────────────
    print("\n[Setting 1] Full-video inference …")
    samples_full = [_make_progress_sample(f, r, max_frames) for f, r in zip(all_frames, valid_rows)]
    preds_full = _run_batched(model, samples_full, args.batch_size, desc="Setting 1")

    results_s1: List[Dict] = []
    for row, frames, preds in zip(valid_rows, all_frames, preds_full):
        target = list(row.get("target_progress") or [])
        gt_last = float(target[-1]) if target else float("nan")
        pred_last = float(preds[-1])
        mae = abs(pred_last - gt_last)
        results_s1.append({
            "id": str(row.get("id", "")),
            "task": row.get("task", ""),
            "num_frames": len(frames),
            "pred_last": pred_last,
            "gt_last": gt_last,
            "mae": mae,
        })

    mae_full = float(np.mean([r["mae"] for r in results_s1]))
    print(f"  mean last-frame MAE (full video): {mae_full:.4f}")

    # ── Setting 2: truncated video ───────────────────────────────────────────
    print(f"\n[Setting 2] Truncated-video inference ({args.n_seeds} seeds/trajectory) …")

    # Collect all truncated samples together for efficient batching
    trunc_samples: List[Any] = []
    trunc_meta: List[Tuple[int, int, int]] = []  # (traj_idx, seed_idx, cutoff)

    for traj_idx, (row, frames) in enumerate(zip(valid_rows, all_frames)):
        N = len(frames)
        lo, hi = max(1, N // 4), N // 2
        if lo >= hi:
            lo = max(1, hi - 1)  # ensure at least one valid cutoff

        rng = np.random.RandomState(args.random_seed + traj_idx * 100)
        cutoffs = rng.randint(lo, hi + 1, size=args.n_seeds).tolist()

        for seed_idx, k in enumerate(cutoffs):
            k = max(1, min(k, N - 1))  # clamp
            trunc_samples.append(_make_progress_sample(frames[:k], row, max_frames))
            trunc_meta.append((traj_idx, seed_idx, k))

    preds_trunc = _run_batched(model, trunc_samples, args.batch_size, desc="Setting 2")

    # Aggregate per-trajectory
    from collections import defaultdict
    traj_seed_results: Dict[int, List[Dict]] = defaultdict(list)

    for (traj_idx, seed_idx, k), preds in zip(trunc_meta, preds_trunc):
        row = valid_rows[traj_idx]
        target = list(row.get("target_progress") or [])
        gt_at_cutoff = float(target[k - 1]) if k <= len(target) else float("nan")
        pred_last = float(preds[-1])
        mae = abs(pred_last - gt_at_cutoff)
        traj_seed_results[traj_idx].append({
            "seed": seed_idx,
            "cutoff": k,
            "pred_last": pred_last,
            "gt_at_cutoff": gt_at_cutoff,
            "mae": mae,
        })

    results_s2: List[Dict] = []
    for traj_idx, (row, frames) in enumerate(zip(valid_rows, all_frames)):
        seed_data = traj_seed_results.get(traj_idx, [])
        mean_mae = float(np.mean([s["mae"] for s in seed_data])) if seed_data else float("nan")
        results_s2.append({
            "id": str(row.get("id", "")),
            "task": row.get("task", ""),
            "num_frames": len(frames),
            "seeds": seed_data,
            "mean_mae": mean_mae,
        })

    mae_truncated = float(np.nanmean([r["mean_mae"] for r in results_s2]))
    print(f"  mean last-frame MAE (truncated):  {mae_truncated:.4f}")

    return {
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "quality_labels": args.quality_labels,
        "n_trajectories": len(valid_rows),
        "n_seeds": args.n_seeds,
        "random_seed": args.random_seed,
        "metrics": {
            "last_frame_mae_full": mae_full,
            "last_frame_mae_truncated": mae_truncated,
        },
        "setting1_full": results_s1,
        "setting2_truncated": results_s2,
    }


# ── output ────────────────────────────────────────────────────────────────────

def save_results(output_dir: str, data: Dict[str, Any]) -> None:
    mae_dir = os.path.join(output_dir, "last_frame_mae")
    Path(mae_dir).mkdir(parents=True, exist_ok=True)

    # Per-trajectory results (one file per quality group to avoid silent overwrites)
    results_out = {
        "setting1_full": data["setting1_full"],
        "setting2_truncated": data["setting2_truncated"],
    }
    quality_labels_for_fname = data.get("quality_labels", [])
    if len(quality_labels_for_fname) == 1:
        results_fname = f"results_{quality_labels_for_fname[0]}.json"
    else:
        results_fname = "results.json"
    with open(os.path.join(mae_dir, results_fname), "w") as f:
        json.dump(_make_json_serializable(results_out), f, indent=2)

    # Eval-level metrics (flat, compatible with combine_checkpoint_metric_plots.py)
    # If a single quality group was evaluated, prefix keys so the combine
    # script can show separate lines per quality in the metric summary plot.
    quality_labels = data.get("quality_labels", [])
    if len(quality_labels) == 1:
        ql = quality_labels[0]
        metrics_flat = {f"{ql}/{k}": v for k, v in data["metrics"].items()}
    else:
        metrics_flat = data["metrics"]
    # Merge with existing metrics (preserves keys from earlier quality runs)
    metrics_path = os.path.join(mae_dir, "metrics.json")
    try:
        with open(metrics_path) as f:
            existing_metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_metrics = {}
    existing_metrics.update(metrics_flat)
    with open(metrics_path, "w") as f:
        json.dump(_make_json_serializable(existing_metrics), f, indent=2)

    # Top-level all_metrics.json (merge with existing if present)
    all_metrics_path = os.path.join(output_dir, "all_metrics.json")
    try:
        with open(all_metrics_path) as f:
            all_metrics = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_metrics = {}
    all_metrics["last_frame_mae"] = existing_metrics  # fully-merged across all quality runs
    with open(all_metrics_path, "w") as f:
        json.dump(_make_json_serializable(all_metrics), f, indent=2)

    print(f"\nSaved results to: {mae_dir}/")
    print(f"  {results_fname}  ({os.path.getsize(os.path.join(mae_dir, results_fname))} bytes)")
    print(f"  metrics.json  → {existing_metrics}")
    print(f"  all_metrics.json updated")


# ── entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join("./last_frame_mae_output", _slug(args.model_path))

    print(f"Output dir : {args.output_dir}")
    print(f"Model      : {args.model_path}")
    print(f"Dataset    : {args.dataset_name}")

    data = run_eval(args)
    save_results(args.output_dir, data)

    print("\nDone.")
    m = data["metrics"]
    print(f"  last_frame_mae_full      = {m['last_frame_mae_full']:.4f}")
    print(f"  last_frame_mae_truncated = {m['last_frame_mae_truncated']:.4f}")


if __name__ == "__main__":
    main()
