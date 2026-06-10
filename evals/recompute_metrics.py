#!/usr/bin/env python3
"""
从已有 results.json 重新计算 reward_alignment 指标并写入 metrics.json / all_metrics.json。

新增：按 quality_label 拆分，key 格式为 "{quality_label}/{metric_name}"。
同时保留无 quality 前缀的 overall 指标（向后兼容）。

跳过 results.json 为空（≤2 bytes）的目录。

Usage:
    cd ~/RoboFAC/robometer
    uv run python evals/recompute_metrics.py \\
        --output-base /home/yingxi/RoboFAC/robometer/baseline_eval_output/all_checkpoint_eval_output_metrics
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-base", required=True,
                   help="包含 checkpoint-N/ / baseline_*/ 子目录的根目录")
    p.add_argument("--dataset-name", default="local_PegInsertionVertical_eval",
                   help="results.json 的数据集名前缀")
    return p.parse_args()


# ── metric computation ───────────────────────────────────────────────────────

def _safe_pearson(xs: List[float], ys: List[float]) -> float:
    try:
        from scipy.stats import pearsonr
        import numpy as np
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
            return float("nan")
        r, _ = pearsonr(xs, ys)
        return float(r) if not math.isnan(r) else float("nan")
    except Exception:
        return float("nan")


def compute_metrics_for_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Per-trajectory pearson (over frame_steps) and MSE loss.

    Each result dict has:
      - id              : trajectory id
      - progress_pred   : list[float], length = num_selected_frames
      - target_progress : list[float], same length
      - metadata.frame_step : int (the last-frame index of this subsequence)

    Pearson is computed over the sequence of (pred[-1], target[-1]) values
    ordered by frame_step within each trajectory, then averaged.
    MSE loss uses the same pred[-1] and target[-1] pairs.
    """
    import numpy as np

    if not results:
        return {}

    # Group by trajectory id
    by_traj: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_traj[str(r.get("id", "unknown"))].append(r)

    pearson_vals, loss_vals = [], []

    for traj_id, traj_results in by_traj.items():
        # Sort by frame_step so we get the temporal ordering
        traj_results.sort(key=lambda r: r.get("metadata", {}).get("frame_step", 0))

        preds   = [float(r["progress_pred"][-1])   for r in traj_results]
        targets = [float(r["target_progress"][-1]) for r in traj_results]

        if len(preds) < 2:
            continue

        p = _safe_pearson(targets, preds)
        if not math.isnan(p):
            pearson_vals.append(p)

        mse = float(np.mean((np.array(preds) - np.array(targets)) ** 2))
        loss_vals.append(mse)

    out: Dict[str, float] = {}
    if pearson_vals:
        out["pearson"] = float(sum(pearson_vals) / len(pearson_vals))
    else:
        out["pearson"] = float("nan")

    if loss_vals:
        out["loss"] = float(sum(loss_vals) / len(loss_vals))
    else:
        out["loss"] = float("nan")

    out["n_trajectories"] = len(by_traj)
    return out


def compute_per_quality_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Returns flat dict with keys "{quality_label}/{metric}" for each quality group
    found in results, plus aggregate "{metric}" without prefix.
    """
    # Group by quality_label
    by_quality: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        ql = str(r.get("quality_label", "unknown"))
        by_quality[ql].append(r)

    flat: Dict[str, float] = {}

    for quality, group in sorted(by_quality.items()):
        m = compute_metrics_for_results(group)
        for k, v in m.items():
            flat[f"{quality}/{k}"] = v

    # Overall across all quality labels
    overall = compute_metrics_for_results(results)
    for k, v in overall.items():
        flat[k] = v

    return flat


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None          # JSON doesn't support NaN; use null
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    return obj


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    base = args.output_base
    dataset_name = args.dataset_name

    run_dirs = sorted(os.listdir(base))
    print(f"Found {len(run_dirs)} run dir(s) under {base}\n")

    for name in run_dirs:
        run_dir = os.path.join(base, name)
        if not os.path.isdir(run_dir):
            continue

        results_file = os.path.join(
            run_dir, "reward_alignment", f"{dataset_name}_results.json"
        )
        if not os.path.exists(results_file):
            print(f"[skip] {name}: no results.json")
            continue
        if os.path.getsize(results_file) <= 2:
            print(f"[skip] {name}: results.json is empty")
            continue

        with open(results_file) as f:
            results = json.load(f)

        if not results:
            print(f"[skip] {name}: results list is empty")
            continue

        n = len(results)
        from collections import Counter
        ql_counts = Counter(r.get("quality_label") for r in results)
        print(f"[recompute] {name}: {n} samples, quality={dict(ql_counts)}", flush=True)

        metrics_out = compute_per_quality_metrics(results)

        # Write reward_alignment/metrics.json
        metrics_file = os.path.join(run_dir, "reward_alignment", "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(_make_serializable(metrics_out), f, indent=2)

        # Write / update all_metrics.json
        all_metrics_path = os.path.join(run_dir, "all_metrics.json")
        try:
            with open(all_metrics_path) as f:
                all_metrics = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_metrics = {}
        all_metrics["reward_alignment"] = metrics_out
        with open(all_metrics_path, "w") as f:
            json.dump(_make_serializable(all_metrics), f, indent=2)

        # Print summary
        for k, v in sorted(metrics_out.items()):
            if v is not None:
                print(f"    {k}: {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
