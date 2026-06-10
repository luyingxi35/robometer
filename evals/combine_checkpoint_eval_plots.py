#!/usr/bin/env python3
"""
Combine per-checkpoint evaluation results into one plot per trajectory.

For each trajectory, creates a single figure with:
  - Top row  : video frames (extracted from the npz file)
  - Bottom   : progress-prediction curves for every checkpoint + the
               Robometer-4B baseline, all on the same axes.

Usage:
    python combine_checkpoint_eval_plots.py \\
        --output-base /data/yingxi/robometer/all_checkpoint_eval_output
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-base",
        required=True,
        help="Directory that contains one checkpoint-N/ subdir per run.",
    )
    parser.add_argument(
        "--combined-dir",
        default=None,
        help="Where to write combined PNGs (default: <output-base>/combined_trajectory_plots/).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json_results(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _find_results_json(run_dir: str, scope: str) -> Optional[str]:
    """Return the first *_results.json inside <run_dir>/<scope>/."""
    scope_dir = os.path.join(run_dir, scope)
    if not os.path.isdir(scope_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(scope_dir, "*_results.json")))
    return candidates[0] if candidates else None


def _group_by_trajectory(
    results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group result dicts by trajectory id, sorted by frame_step within each group."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        traj_id = str(r.get("id", "unknown"))
        grouped[traj_id].append(r)
    for traj_id in grouped:
        grouped[traj_id].sort(
            key=lambda r: r.get("metadata", {}).get("frame_step", 0)
        )
    return dict(grouped)


def _load_frames_from_npz(video_path: str) -> Optional[np.ndarray]:
    """Load frames array from an npz file. Returns None on failure."""
    if not video_path or not os.path.exists(video_path):
        return None
    try:
        from robometer.data.datasets.helpers import load_frames_from_npz
        return load_frames_from_npz(video_path)
    except Exception as exc:
        print(f"  [warn] Could not load frames from {video_path}: {exc}", file=sys.stderr)
        return None


def _frame_for_step(frames: Optional[np.ndarray], step: int) -> Optional[np.ndarray]:
    """Return the frame at `step`, clamped to valid range."""
    if frames is None or len(frames) == 0:
        return None
    idx = min(max(step, 0), len(frames) - 1)
    frame = frames[idx]
    # Normalize to HWC uint8
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _checkpoint_sort_key(name: str) -> int:
    """Extract the numeric step from 'checkpoint-NNN'."""
    m = re.search(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Core plotting
# ---------------------------------------------------------------------------

BASELINE_COLOR = "#ff7f0e"   # orange — matches the original script

# All checkpoints share the Blues colour family; earlier checkpoints are
# lighter, later checkpoints are darker.  We sample from [0.35, 0.92] so
# that even the lightest shade is clearly visible on a white background.
_CKPT_CMAP_LO = 0.35
_CKPT_CMAP_HI = 0.92


def _build_checkpoint_colormap(n: int):
    """Return n shades of blue ordered light→dark (checkpoint-100 → latest)."""
    cmap = cm.get_cmap("Blues")
    if n == 1:
        return [cmap((_CKPT_CMAP_LO + _CKPT_CMAP_HI) / 2)]
    return [cmap(_CKPT_CMAP_LO + (_CKPT_CMAP_HI - _CKPT_CMAP_LO) * i / (n - 1))
            for i in range(n)]


def _plot_combined_trajectory(
    output_path: str,
    trajectory_id: str,
    baseline_rows: List[Dict[str, Any]],
    checkpoint_groups: List[Tuple[str, List[Dict[str, Any]]]],
) -> None:
    """
    Create a combined figure for one trajectory:
      - Top row  : one frame per frame_step
      - Bottom   : progress curves — baseline + all checkpoints

    Parameters
    ----------
    output_path       : where to write the PNG
    trajectory_id     : used as figure title
    baseline_rows     : sorted list of result dicts for the Robometer-4B baseline
    checkpoint_groups : list of (checkpoint_name, sorted_result_dicts) tuples,
                        ordered by checkpoint step number
    """
    # --- gather frame steps (use baseline as the reference) ---
    frame_steps = [
        int(r.get("metadata", {}).get("frame_step", -1)) for r in baseline_rows
    ]

    # --- load video frames ---
    video_path = baseline_rows[0].get("video_path", "")
    frames = _load_frames_from_npz(video_path)
    selected_frames = [_frame_for_step(frames, s) for s in frame_steps]

    # --- build curve data ---
    baseline_curve = [float(r["progress_pred"][-1]) for r in baseline_rows]

    ckpt_colors = _build_checkpoint_colormap(len(checkpoint_groups))
    ckpt_curves: List[Tuple[str, List[float], Any]] = []
    for (ckpt_name, ckpt_rows), color in zip(checkpoint_groups, ckpt_colors):
        # If frame steps differ for this checkpoint, interpolate / align
        ckpt_step_map = {
            int(r.get("metadata", {}).get("frame_step", -1)): float(r["progress_pred"][-1])
            for r in ckpt_rows
        }
        ckpt_curve = [ckpt_step_map.get(s, float("nan")) for s in frame_steps]
        ckpt_curves.append((ckpt_name, ckpt_curve, color))

    # --- figure layout ---
    ncols = max(len(frame_steps), 1)
    fig_width = max(14.0, ncols * 1.8)
    fig = plt.figure(figsize=(fig_width, 9.0))
    grid = fig.add_gridspec(
        2, ncols,
        height_ratios=[1.0, 1.8],
        hspace=0.08,
        wspace=0.04,
    )

    # --- top row: frames ---
    for col, (frame, step) in enumerate(zip(selected_frames, frame_steps)):
        ax = fig.add_subplot(grid[0, col])
        if frame is not None:
            ax.imshow(frame)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"step {step}", fontsize=7)
        ax.axis("off")

    # --- bottom: curves ---
    ax_curve = fig.add_subplot(grid[1, :])
    x = np.arange(len(frame_steps))

    # Draw baseline first (so it sits behind checkpoint lines visually)
    ax_curve.plot(
        x, baseline_curve,
        marker="o", linewidth=2.5, markersize=5,
        color=BASELINE_COLOR,
        label="baseline (Robometer-4B)",
        zorder=5,
    )

    # Draw one line per checkpoint
    for ckpt_name, ckpt_curve, color in ckpt_curves:
        ax_curve.plot(
            x, ckpt_curve,
            marker="^", linewidth=1.8, markersize=4,
            color=color,
            label=ckpt_name,
            alpha=0.85,
        )

    ax_curve.set_xticks(x)
    ax_curve.set_xticklabels([str(s) for s in frame_steps], rotation=45, ha="right")
    ax_curve.set_xlabel("sampled prefix endpoint (frame_step)", fontsize=9)
    ax_curve.set_ylabel("predicted progress", fontsize=9)
    ax_curve.set_ylim(-0.03, 1.03)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(
        loc="best", fontsize=7,
        ncol=max(1, (len(checkpoint_groups) + 2) // 3),
    )
    ax_curve.set_title(f"Trajectory: {trajectory_id}", fontsize=10)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_base = args.output_base
    combined_dir = args.combined_dir or os.path.join(output_base, "combined_trajectory_plots")
    os.makedirs(combined_dir, exist_ok=True)

    # --- discover checkpoint run directories ---
    ckpt_dirs = sorted(
        [
            d for d in glob.glob(os.path.join(output_base, "checkpoint-*"))
            if os.path.isdir(d)
        ],
        key=lambda d: _checkpoint_sort_key(os.path.basename(d)),
    )

    if not ckpt_dirs:
        print(f"No checkpoint-* directories found under {output_base}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ckpt_dirs)} checkpoint run(s):")
    for d in ckpt_dirs:
        print(f"  {os.path.basename(d)}")

    # --- load baseline results (identical across all runs; use first available) ---
    baseline_grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ckpt_dir in ckpt_dirs:
        baseline_json = _find_results_json(ckpt_dir, "baseline")
        if baseline_json:
            print(f"\nLoading baseline results from: {baseline_json}")
            baseline_grouped = _group_by_trajectory(_load_json_results(baseline_json))
            break

    if not baseline_grouped:
        print("Warning: no baseline results found — baseline curve will be omitted.", file=sys.stderr)

    # --- load oracle (checkpoint) results ---
    # List of (checkpoint_name, grouped_results)
    checkpoint_data: List[Tuple[str, Dict[str, List[Dict[str, Any]]]]] = []
    for ckpt_dir in ckpt_dirs:
        ckpt_name = os.path.basename(ckpt_dir)
        oracle_json = _find_results_json(ckpt_dir, "oracle")
        if oracle_json is None:
            print(f"  [warn] No oracle results found for {ckpt_name}, skipping.", file=sys.stderr)
            continue
        grouped = _group_by_trajectory(_load_json_results(oracle_json))
        checkpoint_data.append((ckpt_name, grouped))
        print(f"  Loaded {len(grouped)} trajectories from {ckpt_name}")

    if not checkpoint_data:
        print("No oracle results found in any checkpoint directory.", file=sys.stderr)
        sys.exit(1)

    # --- collect all trajectory ids ---
    all_trajectory_ids = set(baseline_grouped.keys())
    for _, grouped in checkpoint_data:
        all_trajectory_ids.update(grouped.keys())
    all_trajectory_ids = sorted(all_trajectory_ids)

    print(f"\nTotal trajectories to plot: {len(all_trajectory_ids)}")

    # --- create one combined plot per trajectory ---
    for traj_id in all_trajectory_ids:
        output_path = os.path.join(combined_dir, f"{traj_id}.png")

        # Baseline rows for this trajectory
        b_rows = baseline_grouped.get(traj_id)
        if not b_rows:
            # Use empty baseline; need at least one source for frame_steps
            b_rows = []

        # Per-checkpoint rows for this trajectory
        ckpt_groups_for_traj: List[Tuple[str, List[Dict[str, Any]]]] = []
        for ckpt_name, grouped in checkpoint_data:
            rows = grouped.get(traj_id)
            if rows:
                ckpt_groups_for_traj.append((ckpt_name, rows))
            else:
                print(f"  [warn] Trajectory {traj_id} missing from {ckpt_name}", file=sys.stderr)

        # Need at least some rows to determine frame_steps
        if not b_rows and not ckpt_groups_for_traj:
            print(f"  [skip] No data at all for trajectory {traj_id}", file=sys.stderr)
            continue

        # If no baseline rows, borrow frame_steps from first checkpoint for layout
        if not b_rows:
            # Create dummy baseline rows with NaN predictions
            ref_rows = ckpt_groups_for_traj[0][1]
            b_rows = [
                {
                    "id": traj_id,
                    "progress_pred": [float("nan")],
                    "metadata": r.get("metadata", {}),
                    "video_path": r.get("video_path", ""),
                }
                for r in ref_rows
            ]

        _plot_combined_trajectory(
            output_path=output_path,
            trajectory_id=traj_id,
            baseline_rows=b_rows,
            checkpoint_groups=ckpt_groups_for_traj,
        )

    print(f"\nAll combined plots written to: {combined_dir}")


if __name__ == "__main__":
    main()
