#!/usr/bin/env python3
"""
Combine per-checkpoint reward-alignment evaluation results into:

  1. One polished PNG per trajectory showing:
       - Top row   : video frames at each sampled step
       - Bottom    : predicted-progress curves for every checkpoint
                     (Blues palette, light→dark) + Robometer-4B baseline
                     (orange) + ground-truth (grey dashed)

  2. Three metric summary figures — one per data-quality group:
       metric_vs_step_successful_labeled.png
       metric_vs_step_failure_labeled.png
       metric_vs_step_suboptimal_labeled.png
     Each figure contains one subplot per metric (pearson, loss, …)
     showing metric value vs. training step with the Robometer-4B
     baseline as a horizontal dashed reference line.

Usage:
    python combine_checkpoint_metric_plots.py \\
        --output-base ~/RoboFAC/robometer/baseline_eval_output/all_checkpoint_reward_alignment \\
        --baseline-dir ~/RoboFAC/robometer/baseline_eval_output/all_checkpoint_reward_alignment/baseline_Robometer-4B
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
    cmap = mpl.colormaps["Blues"]
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
    """Return the first metrics file found (legacy single-file path)."""
    candidates = [
        os.path.join(run_dir, "all_metrics.json"),
        os.path.join(run_dir, "reward_alignment", "metrics.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _collect_all_metrics(run_dir: str) -> Dict[str, float]:
    """
    Merge metrics from all known sub-eval files into one flat dict.

    Key preservation rules:
    - all_metrics.json      : nested {eval_type: {raw_key: val}} → keep raw_key as-is
                              (raw_key may be "{quality}/{metric}" or "{dataset}/{metric}")
    - reward_alignment/metrics.json / last_frame_mae/metrics.json : flat {raw_key: val}
    Keys that look like "{long_dataset_name}/{metric}" (dataset name has >2 segments or
    contains "eval") are shortened to just "{metric}" for the plot dimension lookup.
    Keys that look like "{quality_label}/{metric}" (quality label is a known short word)
    are kept verbatim so the plotter can group by quality.
    """
    import math as _m

    KNOWN_QUALITY_PREFIXES = {
        "successful_labeled", "successful",
        "failure_labeled", "failure",
        "suboptimal_labeled", "suboptimal",
        "overall",
    }

    def _normalise(key: str, val: Any) -> Optional[tuple]:
        """Return (normalised_key, float_val) or None if not numeric."""
        if not isinstance(val, (int, float)):
            return None
        fval = float(val)
        if _m.isnan(fval):
            return None
        if "/" not in key:
            return key, fval
        prefix, metric = key.rsplit("/", 1)
        # Keep quality-prefix keys verbatim; strip dataset-name prefixes
        if prefix in KNOWN_QUALITY_PREFIXES:
            return key, fval          # e.g. "successful_labeled/pearson"
        return metric, fval           # e.g. "local_PegInsertionVertical_eval/pearson" → "pearson"

    out: Dict[str, float] = {}

    # 1. all_metrics.json  (nested: {eval_type: {raw_key: val}})
    all_path = os.path.join(run_dir, "all_metrics.json")
    if os.path.exists(all_path):
        try:
            raw = _load_json(all_path)
            if isinstance(raw, dict):
                for top_v in raw.values():
                    if isinstance(top_v, dict):
                        for k, v in top_v.items():
                            r = _normalise(k, v)
                            if r:
                                out[r[0]] = r[1]
        except Exception as e:
            print(f"  [warn] {all_path}: {e}", file=sys.stderr)

    # 2. Flat metrics files
    for sub in ("reward_alignment/metrics.json", "last_frame_mae/metrics.json"):
        fpath = os.path.join(run_dir, sub)
        if os.path.exists(fpath):
            try:
                raw = _load_json(fpath)
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        r = _normalise(k, v)
                        if r:
                            out[r[0]] = r[1]
            except Exception as e:
                print(f"  [warn] {fpath}: {e}", file=sys.stderr)

    return out


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

# Display names for the metric part of a "{quality}/{metric}" or "{metric}" key
METRIC_LABELS = {
    "pearson":                  "Pearson Correlation  ↑",
    "loss":                     "Trajectory Loss (MSE)  ↓",
    "last_frame_mae_full":      "Last-frame MAE · Full video  ↓",
    "last_frame_mae_truncated": "Last-frame MAE · Truncated (×5 seeds)  ↓",
    "n_trajectories":           None,   # skip in plots
}
METRIC_YLIMS = {
    "pearson":                  (0.0, 1.05),  # extra headroom so title never overlaps
    "loss":                     (0.0, None),
    "last_frame_mae_full":      (0.0, 1.0),
    "last_frame_mae_truncated": (0.0, 1.0),
}
# Colors per quality label; unknown labels get auto-assigned
QUALITY_PALETTE = {
    "successful_labeled":  "#2a6ebb",
    "successful":          "#2a6ebb",
    "suboptimal_labeled":  "#e6a817",
    "suboptimal":          "#e6a817",
    "failure_labeled":     "#cc3333",
    "failure":             "#cc3333",
    "overall":             "#555555",
}
QUALITY_LINE_STYLES = {
    "successful_labeled":  "-",
    "successful":          "-",
    "suboptimal_labeled":  "--",
    "suboptimal":          "--",
    "failure_labeled":     "-.",
    "failure":             "-.",
    "overall":             ":",
}
# Preferred plot order for metrics (others appended alphabetically)
_METRIC_ORDER = ["pearson", "loss", "last_frame_mae_full", "last_frame_mae_truncated"]
# Preferred quality ordering in legend
_QUALITY_ORDER = ["successful_labeled", "successful",
                  "suboptimal_labeled", "suboptimal",
                  "failure_labeled", "failure", "overall"]


def _parse_metric_key(key: str) -> Tuple[str, str]:
    """Split "{quality}/{metric}" → (quality, metric). No slash → ("overall", key)."""
    if "/" in key:
        quality, metric = key.rsplit("/", 1)
        return quality, metric
    return "overall", key


def _build_quality_color(qualities: List[str]) -> Dict[str, str]:
    """Return color per quality, falling back to a tab10 palette for unknowns."""
    import matplotlib.cm as cm
    colors = dict(QUALITY_PALETTE)
    unknown = [q for q in qualities if q not in colors]
    tab = mpl.colormaps["tab10"]
    for i, q in enumerate(unknown):
        colors[q] = tab((i + 6) % 10)     # offset to avoid clash with our palette
    return colors


def _subplot_one_metric(
    ax,
    metric_name: str,
    quality: str,
    quality_color: Any,
    ckpt_metrics: List[Tuple[int, Dict[str, float]]],
    baseline_metrics: Dict[str, float],
    steps: List[int],
    dot_colors: list,
) -> bool:
    """Draw one metric subplot for one quality group. Returns True if data was found."""
    import math as _math

    raw_key = f"{quality}/{metric_name}"
    vals = [m.get(raw_key) for _, m in ckpt_metrics]
    valid_pts = [(s, v) for s, v in zip(steps, vals)
                 if v is not None and not _math.isnan(v)]

    if not valid_pts:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#aaaaaa", fontsize=11)
        return False

    xs, ys = zip(*valid_pts)
    ls = QUALITY_LINE_STYLES.get(quality, "-")

    ax.plot(xs, ys, linestyle=ls, linewidth=2.2, color=quality_color,
            alpha=0.92, zorder=5)
    for xi, yi, dc in zip(xs, ys, [dot_colors[steps.index(xi)] for xi in xs]):
        ax.scatter(xi, yi, s=60, color=dc, edgecolors=quality_color,
                   linewidth=1.0, zorder=6)
    for xi, yi in zip(xs, ys):
        ax.annotate(f"{yi:.3f}", (xi, yi),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=quality_color)

    # Baseline dashed line — orange, annotated on right margin
    b_val = baseline_metrics.get(raw_key)
    if b_val is None or (isinstance(b_val, float) and _math.isnan(b_val)):
        b_val = baseline_metrics.get(metric_name)  # fallback to overall

    if b_val is not None and not (isinstance(b_val, float) and _math.isnan(b_val)):
        ax.axhline(b_val, linestyle="--", linewidth=1.8,
                   color=BASELINE_COLOR, alpha=0.88, zorder=4)
        ax.annotate(
            f"{b_val:.3f}",
            xy=(1.0, b_val), xycoords=("axes fraction", "data"),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color=BASELINE_COLOR, fontweight="bold",
            va="center", ha="left", annotation_clip=False,
        )

    metric_label = METRIC_LABELS.get(metric_name, metric_name.replace("_", " "))
    ax.set_ylabel(metric_label, fontsize=9, labelpad=6)
    ax.set_xlabel("Training step", fontsize=9)

    if steps:
        ax.set_xticks(steps)
        ax.set_xticklabels([f"{s:,}" for s in steps], rotation=30, ha="right")

    ylim = METRIC_YLIMS.get(metric_name)
    if ylim:
        lo, hi = ylim
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(lo if lo is not None else cur_lo,
                    hi if hi is not None else cur_hi)

    ax.xaxis.grid(True, alpha=0.28, linestyle=":")
    ax.yaxis.grid(True, alpha=0.20)
    return True


def _plot_one_quality(
    out_path: str,
    quality: str,
    unique_metrics: List[str],
    quality_color: Any,
    ckpt_metrics: List[Tuple[int, Dict[str, float]]],
    baseline_metrics: Dict[str, float],
) -> bool:
    """One polished figure for one quality group — subplots for every metric."""
    import math as _math
    from matplotlib.lines import Line2D

    steps = [s for s, _ in ckpt_metrics]
    dot_colors = _ckpt_colors(max(len(steps), 2))
    n = len(unique_metrics)

    # Adaptive grid: ≤2 metrics → 1 row; 3-4 metrics → 2×2
    if n <= 2:
        nrows, ncols = 1, n
    else:
        ncols = 2
        nrows = (n + 1) // 2

    fig_w = max(7.0 * ncols, 10.0)
    fig_h = 5.2 * nrows
    fig, axes_arr = plt.subplots(nrows, ncols,
                                 figsize=(fig_w, fig_h),
                                 gridspec_kw=dict(hspace=0.55, wspace=0.38))
    axes_flat = np.array(axes_arr).flatten().tolist()

    has_any = False
    for ax, metric_name in zip(axes_flat, unique_metrics):
        has_any |= _subplot_one_metric(
            ax=ax, metric_name=metric_name,
            quality=quality, quality_color=quality_color,
            ckpt_metrics=ckpt_metrics, baseline_metrics=baseline_metrics,
            steps=steps, dot_colors=dot_colors,
        )

    # Hide unused axes in the grid
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if not has_any:
        plt.close(fig)
        return False

    # Shared legend at the bottom of the figure
    q_label = quality.replace("_labeled", "").replace("_", " ").title()
    legend_handles = [
        plt.Line2D([0], [0], color=quality_color, linewidth=2.2,
                   linestyle=QUALITY_LINE_STYLES.get(quality, "-"),
                   label=f"Fine-tuned ({q_label})"),
        plt.Line2D([0], [0], color=BASELINE_COLOR, linewidth=1.8,
                   linestyle="--", label="Robometer-4B baseline"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=9, framealpha=0.90,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(f"{q_label}  —  Metric vs. Training Step",
                 fontsize=14, fontweight="bold", y=1.01)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [summary] {out_path}")
    return True


def _plot_one_metric(
    out_path: str,
    metric_name: str,
    unique_qualities: List[str],
    quality_colors: Dict[str, Any],
    ckpt_metrics: List[Tuple[int, Dict[str, float]]],
    baseline_metrics: Dict[str, float],
) -> None:
    """(kept for backward-compat; not called by default any more)"""
    """One polished figure for a single metric — all quality groups on the same axes."""
    import math as _math

    metric_label = METRIC_LABELS.get(metric_name, metric_name.replace("_", " "))
    steps = [s for s, _ in ckpt_metrics]
    dot_colors = _ckpt_colors(max(len(steps), 2))

    fig, ax = plt.subplots(figsize=(10, 5))

    has_any_data = False

    for quality in unique_qualities:
        raw_key = f"{quality}/{metric_name}"
        vals = [m.get(raw_key) for _, m in ckpt_metrics]
        valid_pts = [
            (s, v) for s, v in zip(steps, vals)
            if v is not None and not _math.isnan(v)
        ]
        if not valid_pts:
            continue

        has_any_data = True
        xs, ys = zip(*valid_pts)
        color = quality_colors.get(quality, "#555555")
        ls    = QUALITY_LINE_STYLES.get(quality, "-")
        q_label = quality.replace("_labeled", "").replace("_", " ")

        ax.plot(xs, ys, linestyle=ls, linewidth=2.2, color=color,
                alpha=0.92, zorder=5, label=q_label)
        for xi, yi, dc in zip(xs, ys, [dot_colors[steps.index(xi)] for xi in xs]):
            ax.scatter(xi, yi, s=60, color=dc, edgecolors=color,
                       linewidth=1.0, zorder=6)
        for xi, yi in zip(xs, ys):
            ax.annotate(
                f"{yi:.3f}", (xi, yi),
                xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, color=color,
                fontweight="medium",
            )

    # Baseline — per quality (or fallback to overall)
    plotted_b_qualities = set()
    for quality in unique_qualities:
        b_key = f"{quality}/{metric_name}"
        b_val = baseline_metrics.get(b_key)
        if b_val is None or (isinstance(b_val, float) and _math.isnan(b_val)):
            continue
        color = quality_colors.get(quality, "#555555")
        q_label = quality.replace("_labeled", "").replace("_", " ")
        ax.axhline(b_val, linestyle="--", linewidth=1.6,
                   color=color, alpha=0.55, zorder=3)
        ax.annotate(
            f"baseline ({q_label}): {b_val:.3f}",
            xy=(1.0, b_val), xycoords=("axes fraction", "data"),
            xytext=(7, 0), textcoords="offset points",
            fontsize=7.5, color=color, alpha=0.85,
            va="center", ha="left", annotation_clip=False,
        )
        plotted_b_qualities.add(quality)

    # Fallback overall baseline when no per-quality baseline exists
    if not plotted_b_qualities:
        b_val = baseline_metrics.get(metric_name)
        if b_val is not None and not (isinstance(b_val, float) and _math.isnan(b_val)):
            ax.axhline(b_val, linestyle="--", linewidth=1.8,
                       color=BASELINE_COLOR, alpha=0.88, zorder=3,
                       label=f"Robometer-4B ({b_val:.3f})")
            ax.annotate(
                f"baseline: {b_val:.3f}",
                xy=(1.0, b_val), xycoords=("axes fraction", "data"),
                xytext=(7, 0), textcoords="offset points",
                fontsize=8, color=BASELINE_COLOR, fontweight="bold",
                va="center", ha="left", annotation_clip=False,
            )

    if not has_any_data:
        plt.close(fig)
        print(f"  [skip]    {metric_name}: no data", file=sys.stderr)
        return

    # Axis decoration
    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10, labelpad=8)

    if steps:
        ax.set_xticks(steps)
        ax.set_xticklabels([f"{s:,}" for s in steps])

    ylim = METRIC_YLIMS.get(metric_name)
    if ylim:
        lo, hi = ylim
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(
            lo if lo is not None else cur_lo,
            hi if hi is not None else cur_hi,
        )

    ax.xaxis.grid(True, which="major", alpha=0.28, linestyle=":")
    ax.yaxis.grid(True, which="major", alpha=0.20)

    # Legend inside axes, lower-right (data points are usually in upper region)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.90,
              ncol=min(len(unique_qualities), 3))

    fig.suptitle(metric_label, fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [summary] {out_path}")


def _plot_metric_summary(
    out_dir: str,
    ckpt_metrics: List[Tuple[int, Dict[str, float]]],
    baseline_metrics: Dict[str, float],
) -> List[str]:
    """
    Save ONE PNG per data-quality group into out_dir.
    Each figure shows all available metrics as side-by-side subplots.

    Output filenames:
      metric_vs_step_successful_labeled.png
      metric_vs_step_failure_labeled.png
      metric_vs_step_suboptimal_labeled.png
      (plus any other quality labels found in the data)

    Returns list of saved file paths.
    """
    # ── collect keys ─────────────────────────────────────────────────────────
    all_keys: set = set()
    for _, m in ckpt_metrics:
        all_keys.update(m.keys())
    all_keys.update(baseline_metrics.keys())

    parsed: Dict[str, Tuple[str, str]] = {}
    for key in all_keys:
        q, mn = _parse_metric_key(key)
        parsed[key] = (q, mn)

    skip_metrics = {mn for mn, lbl in METRIC_LABELS.items() if lbl is None}
    unique_metrics = sorted(
        {mn for _, mn in parsed.values() if mn not in skip_metrics},
        key=lambda m: (_METRIC_ORDER.index(m) if m in _METRIC_ORDER else 999, m),
    )

    if not unique_metrics:
        print("  [warn] No numeric metrics to plot.", file=sys.stderr)
        return []

    # One figure per quality group
    unique_qualities = sorted(
        {q for q, _ in parsed.values() if q != "overall"},
        key=lambda q: (_QUALITY_ORDER.index(q) if q in _QUALITY_ORDER else 999, q),
    )
    quality_colors = _build_quality_color(unique_qualities)

    if not unique_qualities:
        print("  [warn] No per-quality metrics found (run recompute_metrics.py first).",
              file=sys.stderr)
        return []

    saved: List[str] = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for quality in unique_qualities:
        out_path = os.path.join(out_dir, f"metric_vs_step_{quality}.png")
        ok = _plot_one_quality(
            out_path=out_path,
            quality=quality,
            unique_metrics=unique_metrics,
            quality_color=quality_colors[quality],
            ckpt_metrics=ckpt_metrics,
            baseline_metrics=baseline_metrics,
        )
        if ok:
            saved.append(out_path)

    return saved


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

    baseline_metrics = _collect_all_metrics(bline_dir)
    if baseline_metrics:
        print(f"Baseline metrics : {baseline_metrics}")
    else:
        print(f"[warn] No metrics found in {bline_dir}", file=sys.stderr)

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

        m = _collect_all_metrics(ckpt_dir)
        if m:
            ckpt_metrics_list.append((step, m))
            print(f"  {name}  metrics: {m}")
        else:
            print(f"  [warn] No metrics for {name}", file=sys.stderr)

    # ── metric summary plots (one PNG per metric) ──
    saved_plots: List[str] = []
    if ckpt_metrics_list:
        saved_plots = _plot_metric_summary(
            out_dir=base,
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
    if saved_plots:
        print(f"  Metric plots ({len(saved_plots)}):")
        for p in saved_plots:
            print(f"    {os.path.basename(p)}")


if __name__ == "__main__":
    main()
