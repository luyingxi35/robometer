#!/usr/bin/env python3
"""
Combine per-checkpoint reward-alignment evaluation results into:

  1. One polished PNG per trajectory showing:
       - Top row   : video frames at each sampled step
       - Bottom    : predicted-progress curves for every checkpoint
                     (Blues palette, light→dark) + Robometer-4B baseline
                     (orange) + ground-truth (grey dashed)

  2. A single summary figure  metric_vs_training_step.png  with one
     subplot per metric (pearson, loss, …) plotting metric value vs.
     checkpoint training step, with the Robometer-4B baseline as a
     horizontal dashed reference line.

Usage:
    python combine_checkpoint_metric_plots.py \\
        --output-base /data/yingxi/robometer/all_checkpoint_eval_output_metrics \\
        --baseline-dir  /data/yingxi/robometer/all_checkpoint_eval_output_metrics/baseline_Robometer-4B
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

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np

# ── global style ────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",
        "figure.dpi": 150,
    }
)

BASELINE_COLOR = "#e07b39"      # warm orange — Robometer-4B
GT_COLOR       = "#6b6b6b"      # mid-grey     — ground truth
GT_LS          = (0, (5, 3))    # custom dashed
_CKPT_LO       = 0.30
_CKPT_HI       = 0.90


# ── helpers ──────────────────────────────────────────────────────────────────

def _ckpt_step(name: str) -> int:
    m = re.search(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else 0


def _ckpt_colors(n: int) -> list:
    cmap = cm.get_cmap("Blues")
    if n == 1:
        return [cmap((_CKPT_LO + _CKPT_HI) / 2)]
    return [cmap(_CKPT_LO + (_CKPT_HI - _CKPT_LO) * i / (n - 1)) for i in range(n)]


def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _find_results_json(run_dir: str) -> Optional[str]:
    """Return the first reward_alignment/*_results.json inside run_dir."""
    candidates = sorted(
        glob.glob(os.path.join(run_dir, "reward_alignment", "*_results.json"))
    )
    return candidates[0] if candidates else None


def _find_metrics_json(run_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(run_dir, "all_metrics.json"),
        os.path.join(run_dir, "reward_alignment", "metrics.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _group_by_traj(results: List[Dict]) -> Dict[str, List[Dict]]:
    g: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        g[str(r.get("id", "unknown"))].append(r)
    for tid in g:
        g[tid].sort(key=lambda r: r.get("metadata", {}).get("frame_step", 0))
    return dict(g)


def _load_frames(video_path: str) -> Optional[np.ndarray]:
    if not video_path or not os.path.exists(video_path):
        return None
    try:
        from robometer.data.datasets.helpers import load_frames_from_npz
        return load_frames_from_npz(video_path)
    except Exception as e:
        print(f"  [warn] frames load failed ({e})", file=sys.stderr)
        return None


def _pick_frame(frames: Optional[np.ndarray], step: int) -> Optional[np.ndarray]:
    if frames is None or len(frames) == 0:
        return None
    f = frames[min(max(step, 0), len(frames) - 1)]
    if f.ndim == 3 and f.shape[0] in (1, 3):
        f = np.transpose(f, (1, 2, 0))
    if f.dtype != np.uint8:
        f = (np.clip(f, 0.0, 1.0) * 255).astype(np.uint8) if f.max() <= 1.0 else np.clip(f, 0, 255).astype(np.uint8)
    return f


def _extract_scalar_metrics(metrics_raw: Any) -> Dict[str, float]:
    """Flatten nested metrics dict to {metric_name: float}."""
    out: Dict[str, float] = {}
    if isinstance(metrics_raw, dict):
        for k, v in metrics_raw.items():
            if isinstance(v, dict):
                # nested: {"reward_alignment": {"local_eval/.../pearson": 0.5}}
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        # keep only the metric name (last segment after last '/')
                        metric_name = k2.split("/")[-1]
                        out[metric_name] = float(v2)
            elif isinstance(v, (int, float)):
                metric_name = k.split("/")[-1]
                out[metric_name] = float(v)
    return out


# ── per-trajectory plot ───────────────────────────────────────────────────────

def _plot_trajectory(
    out_path: str,
    traj_id: str,
    baseline_rows: List[Dict],
    ckpt_groups: List[Tuple[str, List[Dict]]],    # [(name, rows), …]  sorted by step
) -> None:
    """One polished PNG per trajectory."""
    ref_rows = baseline_rows or ckpt_groups[0][1]
    frame_steps = [int(r.get("metadata", {}).get("frame_step", -1)) for r in ref_rows]
    n = len(frame_steps)

    # ── load video frames ──
    video_path = ref_rows[0].get("video_path", "")
    frames = _load_frames(video_path)
    imgs = [_pick_frame(frames, s) for s in frame_steps]

    # ── build curves ──
    def _curve(rows: List[Dict]) -> List[float]:
        step_map = {
            int(r.get("metadata", {}).get("frame_step", -1)): float(r["progress_pred"][-1])
            for r in rows
        }
        return [step_map.get(s, float("nan")) for s in frame_steps]

    # GT from baseline rows (should be identical across checkpoints)
    def _gt(rows: List[Dict]) -> List[float]:
        step_map = {}
        for r in rows:
            fs = int(r.get("metadata", {}).get("frame_step", -1))
            tp = r.get("target_progress")
            if tp is not None:
                step_map[fs] = float(np.asarray(tp).flat[-1])
        return [step_map.get(s, float("nan")) for s in frame_steps]

    baseline_curve = _curve(baseline_rows) if baseline_rows else []
    gt_curve       = _gt(baseline_rows or ckpt_groups[0][1])
    colors         = _ckpt_colors(len(ckpt_groups))

    # ── layout ──
    ncols     = max(n, 1)
    fig_w     = max(14.0, ncols * 1.9)
    fig, axes = plt.subplots(
        2, ncols,
        figsize=(fig_w, 9),
        gridspec_kw=dict(height_ratios=[1, 2], hspace=0.06, wspace=0.04),
    )
    if ncols == 1:
        axes = axes.reshape(2, 1)

    # ── frame row ──
    for col, (img, step) in enumerate(zip(imgs, frame_steps)):
        ax = axes[0, col]
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor("#f0f0f0")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, color="#999999", fontsize=8)
        ax.set_title(f"t = {step}", fontsize=7, pad=3, color="#444444")
        ax.axis("off")

    # ── curve subplot ──
    ax_c = fig.add_subplot(axes[1, 0].get_gridspec()[1, :])
    for a in axes[1]:
        a.remove()

    x = np.arange(n)

    # GT (drawn first so it sits behind everything)
    has_gt = any(not np.isnan(v) for v in gt_curve)
    if has_gt:
        ax_c.plot(
            x, gt_curve,
            linestyle=GT_LS, linewidth=2.0, marker="D", markersize=4,
            color=GT_COLOR, label="ground truth",
            alpha=0.75, zorder=4,
        )

    # Checkpoint curves (light → dark blue)
    for (ckpt_name, ckpt_rows), col in zip(ckpt_groups, colors):
        step_n = _ckpt_step(ckpt_name)
        ax_c.plot(
            x, _curve(ckpt_rows),
            linestyle="-", linewidth=1.6, marker="^", markersize=4,
            color=col, label=f"ckpt {step_n:,}", alpha=0.90, zorder=5,
        )

    # Baseline (drawn on top)
    if baseline_curve:
        ax_c.plot(
            x, baseline_curve,
            linestyle="-", linewidth=2.2, marker="o", markersize=5,
            color=BASELINE_COLOR, label="Robometer-4B (baseline)",
            zorder=6,
        )

    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(s) for s in frame_steps], rotation=40, ha="right")
    ax_c.set_xlabel("prefix length (frame step)", labelpad=6)
    ax_c.set_ylabel("predicted progress")
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_c.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    # custom legend: put GT at bottom of entry list for visual clarity
    handles, labels = ax_c.get_legend_handles_labels()
    ncol = max(1, (len(handles) + 1) // 2)
    ax_c.legend(handles, labels, loc="upper left", ncol=ncol,
                frameon=True, borderpad=0.6, handlelength=2.0)

    quality = (baseline_rows or ckpt_groups[0][1])[0].get("quality_label", "")
    task    = (baseline_rows or ckpt_groups[0][1])[0].get("task", "")
    subtitle = f"{task}  ·  {quality}" if task else quality
    fig.suptitle(
        f"Trajectory  {traj_id}" + (f"\n{subtitle}" if subtitle else ""),
        fontsize=11, fontweight="bold", y=0.98,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [traj] {out_path}")


# ── metric summary plot ───────────────────────────────────────────────────────

METRIC_LABELS = {
    "pearson": "Pearson Correlation  ↑",
    "loss":    "Trajectory Loss  ↓",
}
METRIC_YLIMS = {
    "pearson": (0, 1),
    "loss":    None,          # auto
}


def _plot_metric_summary(
    out_path: str,
    ckpt_metrics: List[Tuple[int, Dict[str, float]]],   # [(step, metrics), …]  sorted
    baseline_metrics: Dict[str, float],
) -> None:
    """Metric value vs. training step — one subplot per metric key."""
    all_keys = sorted(
        {k for _, m in ckpt_metrics for k in m}
        | set(baseline_metrics.keys())
    )
    # Only plot numeric metrics we recognise
    plot_keys = [k for k in ("pearson", "loss") if k in all_keys]
    extra = [k for k in all_keys if k not in ("pearson", "loss")
             and not k.startswith("_")]
    plot_keys += extra

    if not plot_keys:
        print("  [warn] No numeric metrics to plot.", file=sys.stderr)
        return

    n_metrics  = len(plot_keys)
    fig_h      = 3.6 * n_metrics
    fig, axes  = plt.subplots(n_metrics, 1, figsize=(9, fig_h),
                               gridspec_kw=dict(hspace=0.55))
    if n_metrics == 1:
        axes = [axes]

    steps  = [s for s, _ in ckpt_metrics]
    colors = _ckpt_colors(max(len(steps), 2))   # reuse same palette for dots

    for ax, key in zip(axes, plot_keys):
        vals = [m.get(key, float("nan")) for _, m in ckpt_metrics]
        b_val = baseline_metrics.get(key)

        # ── checkpoint curve ──
        valid = [(s, v) for s, v in zip(steps, vals) if not np.isnan(v)]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, linestyle="-", linewidth=2.0,
                    color="#2a6ebb", zorder=5, alpha=0.9)
            for xi, yi, ci in zip(xs, ys, [colors[steps.index(x)] for x in xs]):
                ax.scatter(xi, yi, s=55, color=ci, edgecolors="#1a4e8a",
                           linewidth=0.8, zorder=6)
            # annotate each point
            for xi, yi in zip(xs, ys):
                ax.annotate(
                    f"{yi:.3f}", (xi, yi),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="#2a6ebb",
                )

        # ── baseline dashed line ──
        if b_val is not None and not np.isnan(b_val):
            x_lo = min(steps) * 0.85 if steps else 0
            x_hi = max(steps) * 1.10 if steps else 1
            ax.axhline(b_val, linestyle="--", linewidth=1.8,
                       color=BASELINE_COLOR, alpha=0.90, zorder=4)
            ax.annotate(
                f"Robometer-4B  {b_val:.3f}",
                xy=(x_lo + (x_hi - x_lo) * 0.02, b_val),
                xytext=(0, 5), textcoords="offset points",
                fontsize=8, color=BASELINE_COLOR, fontweight="bold",
            )

        # ── axis decoration ──
        label = METRIC_LABELS.get(key, key)
        ax.set_ylabel(label)
        ax.set_xlabel("training step")
        ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
        if steps:
            ax.set_xticks(steps)
            ax.set_xticklabels([f"{s:,}" for s in steps])
        ylim = METRIC_YLIMS.get(key)
        if ylim:
            ax.set_ylim(*ylim)

        # light vertical gridlines at each step
        ax.xaxis.grid(True, which="major", alpha=0.35, linestyle=":")
        ax.yaxis.grid(True, which="major", alpha=0.25)

        # custom legend entry for baseline
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="#2a6ebb", linewidth=2, label="fine-tuned ckpt"),
            Line2D([0], [0], color=BASELINE_COLOR, linewidth=1.8,
                   linestyle="--", label="Robometer-4B baseline"),
        ]
        ax.legend(handles=legend_handles, loc="best")

    fig.suptitle("Metric vs. Training Step", fontsize=13, fontweight="bold", y=1.01)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [summary] {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-base", required=True,
                   help="Dir containing one checkpoint-N/ per run.")
    p.add_argument("--baseline-dir", default=None,
                   help="Dir of Robometer-4B eval. Defaults to "
                        "<output-base>/baseline_Robometer-4B/")
    p.add_argument("--combined-dir", default=None,
                   help="Output for trajectory PNGs (default: "
                        "<output-base>/combined_trajectory_plots/)")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    base      = args.output_base
    bline_dir = args.baseline_dir or os.path.join(base, "baseline_Robometer-4B")
    traj_dir  = args.combined_dir or os.path.join(base, "combined_trajectory_plots")
    os.makedirs(traj_dir, exist_ok=True)

    # ── discover checkpoint runs ──
    ckpt_dirs = sorted(
        [d for d in glob.glob(os.path.join(base, "checkpoint-*")) if os.path.isdir(d)],
        key=lambda d: _ckpt_step(os.path.basename(d)),
    )
    if not ckpt_dirs:
        print(f"No checkpoint-* dirs under {base}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ckpt_dirs)} checkpoint run(s):")
    for d in ckpt_dirs:
        print(f"  {os.path.basename(d)}")

    # ── load baseline ──
    baseline_grouped: Dict[str, List[Dict]] = {}
    baseline_metrics: Dict[str, float]      = {}

    bline_json = _find_results_json(bline_dir)
    if bline_json:
        print(f"\nBaseline results : {bline_json}")
        baseline_grouped = _group_by_traj(_load_json(bline_json))
    else:
        print(f"[warn] No baseline results in {bline_dir}", file=sys.stderr)

    bline_m = _find_metrics_json(bline_dir)
    if bline_m:
        baseline_metrics = _extract_scalar_metrics(_load_json(bline_m))
        print(f"Baseline metrics : {baseline_metrics}")

    # ── load checkpoint results & metrics ──
    ckpt_data: List[Tuple[str, Dict[str, List[Dict]]]] = []
    ckpt_metrics_list: List[Tuple[int, Dict[str, float]]] = []

    for ckpt_dir in ckpt_dirs:
        name  = os.path.basename(ckpt_dir)
        step  = _ckpt_step(name)

        rj = _find_results_json(ckpt_dir)
        if rj:
            ckpt_data.append((name, _group_by_traj(_load_json(rj))))
        else:
            print(f"  [warn] No results for {name}", file=sys.stderr)

        mj = _find_metrics_json(ckpt_dir)
        if mj:
            m = _extract_scalar_metrics(_load_json(mj))
            ckpt_metrics_list.append((step, m))
            print(f"  {name}  metrics: {m}")
        else:
            print(f"  [warn] No metrics for {name}", file=sys.stderr)

    # ── metric summary plot ──
    if ckpt_metrics_list:
        _plot_metric_summary(
            out_path=os.path.join(base, "metric_vs_training_step.png"),
            ckpt_metrics=ckpt_metrics_list,
            baseline_metrics=baseline_metrics,
        )

    # ── per-trajectory plots ──
    all_traj_ids = sorted(
        set(baseline_grouped.keys()) | {tid for _, g in ckpt_data for tid in g}
    )
    print(f"\nTrajectories to plot: {len(all_traj_ids)}")

    for tid in all_traj_ids:
        b_rows = baseline_grouped.get(tid, [])

        ckpt_groups = [
            (name, grouped[tid])
            for name, grouped in ckpt_data
            if tid in grouped
        ]
        if not b_rows and not ckpt_groups:
            print(f"  [skip] {tid} — no data at all", file=sys.stderr)
            continue

        # borrow frame layout from first checkpoint if baseline missing
        if not b_rows and ckpt_groups:
            ref = ckpt_groups[0][1]
            b_rows = [
                {
                    **r,
                    "progress_pred": [float("nan")],
                    "target_progress": None,
                }
                for r in ref
            ]

        _plot_trajectory(
            out_path=os.path.join(traj_dir, f"{tid}.png"),
            traj_id=tid,
            baseline_rows=b_rows,
            ckpt_groups=ckpt_groups,
        )

    print(f"\nAll done.")
    print(f"  Trajectory plots : {traj_dir}/")
    print(f"  Metric summary   : {base}/metric_vs_training_step.png")


if __name__ == "__main__":
    main()
