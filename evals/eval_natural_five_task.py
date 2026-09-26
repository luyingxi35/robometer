#!/usr/bin/env python3
"""Natural five-task checkpoint evaluation.

Computes progress MAE/Spearman, success-head failure detection, and trajectory
filtering metrics for one checkpoint or a sequential sweep.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

os.environ.setdefault("ROBOMETER_PROCESSED_DATASETS_PATH", "/data/yingxi/robometer")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

TASK_RE = re.compile(r"^(?P<task>.+?)_(?:test|success|failure|suboptimal)_")
QUALITY_LABELS = ("successful_labeled", "failure_labeled")
# Filtering uses a single progress-only composite score. Components are
# normalized per task over the candidate trajectories before equal weighting.
SCORE_METHODS = ("pearson_delta_final_minmax",)
FILTER_SCORE_WEIGHTS = {"pearson": 1.0, "delta": 1.0, "final": 1.0}
ROOT = Path("/data/yingxi/robometer/natural_five_task_20260921")
FORMAL_DATASET = ROOT / "processed/local_natural_five_task_test/processed_dataset"
FORMAL_MANIFEST = ROOT / "local_hf/natural_five_task_test/selection_manifest.json"
SMOKE_DATASET = ROOT / "smoke_processed/local_natural_five_task_test_smoke/processed_dataset"
SMOKE_MANIFEST = ROOT / "smoke_local_hf/natural_five_task_test/selection_manifest.json"
PILOT_CKPT_DIR = ROOT / "training/natural_five_task_fsdp_pilot"
SMOKE_CKPT = ROOT / "training/natural_five_task_fsdp_smoke_v2/checkpoint-10"


def json_ready(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(json_ready(data), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def step_from_checkpoint(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", str(path.rstrip("/") if isinstance(path, str) else path))
    if not m:
        m = re.search(r"checkpoint-(\d+)$", path.name)
    if not m:
        raise ValueError(f"Cannot infer step from checkpoint path: {path}")
    return int(m.group(1))


def discover_checkpoints(checkpoint_dir: Path) -> List[Path]:
    checkpoints = []
    for path in checkpoint_dir.glob("checkpoint-*"):
        if path.is_dir():
            try:
                step_from_checkpoint(path)
            except ValueError:
                continue
            checkpoints.append(path)
    return sorted(checkpoints, key=step_from_checkpoint)


def task_scope(row: Dict[str, Any]) -> str:
    rid = str(row.get("id", ""))
    m = TASK_RE.match(rid)
    if m:
        return m.group("task")
    return str(row.get("task", "")).strip() or "unknown"


def load_rows(dataset_path: Path, per_class: Optional[int]) -> List[Dict[str, Any]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(dataset_path))
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in ds:
        label = row.get("quality_label")
        if label not in QUALITY_LABELS:
            continue
        buckets[(task_scope(row), label)].append(row)

    rows = []
    for key in sorted(buckets):
        chunk = buckets[key]
        rows.extend(chunk[:per_class] if per_class else chunk)
    return rows


def subsample_frames(frames: np.ndarray, max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(frames)
    if n <= max_frames:
        return frames, np.arange(n)
    idx = np.round(np.linspace(0, n - 1, max_frames)).astype(int)
    return frames[idx], idx


def make_sample(frames: np.ndarray, row: Dict[str, Any], max_frames: int):
    from robometer.data.dataset_types import ProgressSample, Trajectory

    subsampled, idx = subsample_frames(frames, max_frames)
    target = list(row.get("target_progress") or [])
    target_sub = [float(target[i]) if i < len(target) else 0.0 for i in idx]
    traj = Trajectory(
        frames=subsampled,
        frames_shape=tuple(subsampled.shape),
        target_progress=target_sub,
        predict_last_frame_mask=[1.0] * len(subsampled),
        success_label=[1.0 if row.get("quality_label") == "successful_labeled" else 0.0] * len(subsampled),
        task=row.get("task", ""),
        id=str(row.get("id", "")),
        quality_label=row.get("quality_label", ""),
        data_source=row.get("data_source", ""),
        partial_success=row.get("partial_success"),
    )
    return ProgressSample(trajectory=traj, sample_type="progress"), target_sub, idx.tolist()


def ranks(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    order = np.argsort(arr)
    out = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        out[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return out


def safe_corr(a: Iterable[float], b: Iterable[float], ranked: bool = False) -> Optional[float]:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None
    if ranked:
        x, y = ranks(x), ranks(y)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def auc_roc(labels: List[int], scores: List[float]) -> Optional[float]:
    pos = [(s, y) for y, s in zip(labels, scores) if y == 1]
    neg = [(s, y) for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps, _ in pos:
        for ns, _ in neg:
            wins += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
    return wins / (len(pos) * len(neg))


def auc_pr(labels: List[int], scores: List[float]) -> Optional[float]:
    if not any(labels):
        return None
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    prev_recall = 0.0
    area = 0.0
    total_pos = sum(labels)
    for _, y in pairs:
        if y:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / max(1, tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def pearson_failure_stats(progress: Iterable[float], window: int, threshold: float) -> Tuple[bool, Optional[float]]:
    values = np.asarray(list(progress), dtype=float)
    if len(values) < window:
        return False, None
    x = np.arange(window, dtype=float)
    correlations = []
    for start in range(len(values) - window + 1):
        y = values[start : start + window]
        corr = 0.0 if float(np.std(y)) < 1e-8 else float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(corr):
            correlations.append(corr)
    if not correlations:
        return False, None
    minimum = min(correlations)
    return minimum <= threshold, minimum


def failure_metrics(records: List[Dict[str, Any]], pearson_window: int, pearson_threshold: float) -> Dict[str, Any]:
    labels = [1 if r["quality_label"] == "failure_labeled" else 0 for r in records]
    stats = [pearson_failure_stats(r["pred_progress"], pearson_window, pearson_threshold) for r in records]
    scores = [float(-s[1]) if s[1] is not None else 0.0 for s in stats]
    preds = [1 if s[0] else 0 for s in stats]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    success_rec = spec
    macro_f1 = (f1 + (2 * success_rec * (tn / (tn + fn) if tn + fn else 0.0) / (success_rec + (tn / (tn + fn) if tn + fn else 0.0)) if success_rec + (tn / (tn + fn) if tn + fn else 0.0) else 0.0)) / 2
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": len(records),
        "score_source": "minimum_sliding_window_progress_pearson",
        "threshold_rule": f"failure if any {pearson_window}-frame progress Pearson <= {pearson_threshold}",
        "accuracy": (tp + tn) / len(records) if records else 0.0,
        "macro_f1": macro_f1,
        "balanced_accuracy": (rec + spec) / 2,
        "mcc": ((tp * tn - fp * fn) / denom) if denom else 0.0,
        "auroc": auc_roc(labels, scores),
        "auprc": auc_pr(labels, scores),
        "failure_precision": prec,
        "failure_recall": rec,
        "failure_f1": f1,
        "success_specificity": spec,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def score_record(record: Dict[str, Any], method: str) -> float:
    pred = np.asarray(record["pred_progress"], dtype=float)
    gt = np.asarray(record["target_progress"], dtype=float)
    if method == "pearson":
        return safe_corr(pred, gt) or -1.0
    if method == "final":
        return float(pred[-1])
    if method == "delta":
        return float(pred[-1] - pred[0])
    if method == "slope":
        if len(pred) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(pred)), pred, 1)[0])
    if method == "late_mean_delta":
        mid = max(1, len(pred) // 2)
        return float(np.mean(pred[mid:]) - np.mean(pred[:mid]))
    raise ValueError(method)


def filtering_component_scores(records: List[Dict[str, Any]]) -> np.ndarray:
    """Return per-task min-max normalized [Pearson, delta, final] scores."""
    components = []
    for record in records:
        pred = np.asarray(record["pred_progress"], dtype=float)
        target = np.asarray(record["target_progress"], dtype=float)
        pearson = safe_corr(pred, target)
        components.append([
            -1.0 if pearson is None else float(pearson),
            float(pred[-1] - pred[0]),
            float(pred[-1]),
        ])
    values = np.asarray(components, dtype=float)
    if not len(values):
        return values
    low = values.min(axis=0)
    high = values.max(axis=0)
    span = high - low
    return np.divide(values - low, span, out=np.zeros_like(values), where=span > 1e-12)


def filtering_composite_scores(records: List[Dict[str, Any]]) -> np.ndarray:
    normalized = filtering_component_scores(records)
    if not len(normalized):
        return np.asarray([], dtype=float)
    weights = np.asarray([FILTER_SCORE_WEIGHTS[name] for name in ("pearson", "delta", "final")], dtype=float)
    return normalized @ weights


def build_metrics(records: List[Dict[str, Any]], step: int, model_path: str, top_k: int, pearson_window: int, pearson_threshold: float) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    tasks = sorted({r["scope"] for r in records})

    def with_common(row: Dict[str, Any]) -> Dict[str, Any]:
        row.update({"step": step, "model_path": model_path})
        return row

    progress_rows = []
    for task in tasks:
        for label in (*QUALITY_LABELS, "all"):
            subset = [r for r in records if r["scope"] == task and (label == "all" or r["quality_label"] == label)]
            if not subset:
                continue
            maes = [float(np.mean(np.abs(np.asarray(r["pred_progress"]) - np.asarray(r["target_progress"])))) for r in subset]
            spears = [safe_corr(r["pred_progress"], r["target_progress"], ranked=True) for r in subset]
            valid = [s for s in spears if s is not None]
            row = with_common({
                "family": "progress",
                "scope": task,
                "quality_label": label,
                "n": len(subset),
                "mae": float(np.mean(maes)),
                "spearman": float(np.mean(valid)) if valid else None,
                "valid_spearman_n": len(valid),
                "invalid_spearman_n": len(spears) - len(valid),
            })
            progress_rows.append(row)
            metrics.append(row)
    all_subset = records
    if all_subset:
        maes = [float(np.mean(np.abs(np.asarray(r["pred_progress"]) - np.asarray(r["target_progress"])))) for r in all_subset]
        spears = [safe_corr(r["pred_progress"], r["target_progress"], ranked=True) for r in all_subset]
        valid = [s for s in spears if s is not None]
        metrics.append(with_common({
            "family": "progress",
            "scope": "sample_micro",
            "quality_label": "all",
            "n": len(all_subset),
            "mae": float(np.mean(maes)),
            "spearman": float(np.mean(valid)) if valid else None,
            "valid_spearman_n": len(valid),
            "invalid_spearman_n": len(spears) - len(valid),
        }))
    task_all = [r for r in progress_rows if r["quality_label"] == "all"]
    if task_all:
        metrics.append(with_common({
            "family": "progress",
            "scope": "task_macro",
            "quality_label": "all",
            "n": int(sum(r["n"] for r in task_all)),
            "mae": float(np.mean([r["mae"] for r in task_all])),
            "spearman": float(np.mean([r["spearman"] for r in task_all if r["spearman"] is not None])),
            "valid_spearman_n": int(sum(r["valid_spearman_n"] for r in task_all)),
            "invalid_spearman_n": int(sum(r["invalid_spearman_n"] for r in task_all)),
        }))

    fail_rows = []
    for task in tasks:
        subset = [r for r in records if r["scope"] == task]
        row = with_common({"family": "failure_detection", "scope": task, **failure_metrics(subset, pearson_window, pearson_threshold)})
        fail_rows.append(row)
        metrics.append(row)
    if fail_rows:
        keys = ["accuracy", "macro_f1", "balanced_accuracy", "mcc", "failure_precision", "failure_recall", "failure_f1", "success_specificity"]
        metrics.append(with_common({
            "family": "failure_detection",
            "scope": "task_macro",
            "n": int(sum(r["n"] for r in fail_rows)),
            **{k: float(np.mean([r[k] for r in fail_rows])) for k in keys},
            "auroc": float(np.mean([r["auroc"] for r in fail_rows if r["auroc"] is not None])),
            "auprc": float(np.mean([r["auprc"] for r in fail_rows if r["auprc"] is not None])),
            "tp": int(sum(r["tp"] for r in fail_rows)),
            "tn": int(sum(r["tn"] for r in fail_rows)),
            "fp": int(sum(r["fp"] for r in fail_rows)),
            "fn": int(sum(r["fn"] for r in fail_rows)),
        }))
    metrics.append(with_common({"family": "failure_detection", "scope": "sample_micro", **failure_metrics(records, pearson_window, pearson_threshold)}))

    filtering_rows = []
    for task in tasks:
        subset = [r for r in records if r["scope"] == task]
        k = min(top_k, len(subset))
        composite = filtering_composite_scores(subset)
        ranked = [subset[i] for i in np.argsort(-composite, kind="stable")[:k]]
        best = sum(1 for r in ranked if r["quality_label"] == "successful_labeled")
        row = with_common({
            "family": "filtering",
            "scope": task,
            "score_method": SCORE_METHODS[0],
            "score_components": ["pearson", "delta", "final"],
            "normalization": "per_task_minmax",
            "score_weights": FILTER_SCORE_WEIGHTS,
            "candidate_count": len(subset),
            "top_k": k,
            "correct_count": best,
            "correctness": best / k if k else 0.0,
            "best_count": best,
            "best_rate": best / k if k else 0.0,
            "success_total": sum(1 for r in subset if r["quality_label"] == "successful_labeled"),
        })
        filtering_rows.append(row)
        metrics.append(row)
    for method in SCORE_METHODS:
        rows = [r for r in filtering_rows if r["score_method"] == method]
        metrics.append(with_common({
            "family": "filtering",
            "scope": "task_macro",
            "score_method": method,
            "score_components": ["pearson", "delta", "final"],
            "normalization": "per_task_minmax",
            "score_weights": FILTER_SCORE_WEIGHTS,
            "candidate_count": int(sum(r["candidate_count"] for r in rows)),
            "top_k": int(sum(r["top_k"] for r in rows)),
            "correct_count": int(sum(r["correct_count"] for r in rows)),
            "correctness": float(np.mean([r["correctness"] for r in rows])) if rows else 0.0,
            "best_count": int(sum(r["best_count"] for r in rows)),
            "best_rate": float(np.mean([r["best_rate"] for r in rows])) if rows else 0.0,
            "success_total": int(sum(r["success_total"] for r in rows)),
        }))
    return metrics


def run_checkpoint(args: argparse.Namespace, checkpoint: Path, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from robometer.data.datasets.helpers import load_frames_from_npz
    from robometer.evals.baselines.rbm_model import RBMModel

    step = step_from_checkpoint(checkpoint)
    step_metrics = args.output_dir / f"step_{step:06d}_metrics.json"
    if step_metrics.exists() and not args.force:
        print(f"[skip] step {step}: {step_metrics} exists")
        with step_metrics.open() as f:
            return json.load(f)

    print(f"[eval] step {step}: {checkpoint}")
    model = RBMModel(str(checkpoint))
    samples = []
    meta = []
    for row in tqdm(rows, desc="Load frames"):
        frames_path = row.get("frames")
        if not isinstance(frames_path, str) or not os.path.exists(frames_path):
            print(f"[skip] missing frames: {row.get('id')} {frames_path}", file=sys.stderr)
            continue
        frames = load_frames_from_npz(frames_path)
        if len(frames) < 2:
            continue
        sample, target_sub, indices = make_sample(frames, row, args.max_frames)
        samples.append(sample)
        meta.append((row, target_sub, indices))

    progress_preds: List[List[float]] = []
    success_preds: List[List[float]] = []
    for start in tqdm(range(0, len(samples), args.batch_size), desc=f"Inference step {step}"):
        batch = samples[start:start + args.batch_size]
        p, s = model.compute_batched_progress_and_success(batch)
        progress_preds.extend(p)
        success_preds.extend(s)

    records = []
    for (row, target_sub, indices), pred, success in zip(meta, progress_preds, success_preds):
        n = min(len(pred), len(target_sub))
        records.append({
            "id": str(row.get("id", "")),
            "scope": task_scope(row),
            "task": row.get("task", ""),
            "quality_label": row.get("quality_label", ""),
            "frame_indices": indices[:n],
            "target_progress": [float(x) for x in target_sub[:n]],
            "pred_progress": [float(x) for x in pred[:n]],
            "success_prob": float(success[min(len(success), n) - 1]) if success and n else 0.0,
        })

    metrics = build_metrics(records, step, str(checkpoint), args.top_k, args.pearson_window, args.pearson_threshold)
    atomic_json(step_metrics, metrics)
    atomic_json(args.output_dir / f"step_{step:06d}_records.json", records)
    return metrics


def write_run_config(args: argparse.Namespace, checkpoints: List[Path]) -> None:
    atomic_json(args.output_dir / "run_config.json", {
        "mode": args.mode,
        "code_commit": git_commit(),
        "dataset_path": args.dataset_path,
        "selection_manifest": args.selection_manifest,
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "per_class": args.per_class,
        "top_k": args.top_k,
        "filtering_score_method": SCORE_METHODS[0],
        "filtering_score_components": ["pearson", "delta", "final"],
        "filtering_normalization": "per_task_minmax",
        "filtering_score_weights": FILTER_SCORE_WEIGHTS,
        "pearson_window": args.pearson_window,
        "pearson_threshold": args.pearson_threshold,
        "batch_size": args.batch_size,
        "max_frames": args.max_frames,
        "checkpoints": [{"step": step_from_checkpoint(p), "path": p} for p in checkpoints],
    })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp):
        sp.add_argument("--dataset-path", type=Path, default=FORMAL_DATASET)
        sp.add_argument("--selection-manifest", type=Path, default=FORMAL_MANIFEST)
        sp.add_argument("--output-dir", type=Path, default=ROOT / "eval_sweep")
        sp.add_argument("--per-class", type=int, default=50)
        sp.add_argument("--top-k", type=int, default=50)
        sp.add_argument("--pearson-window", type=int, default=5)
        sp.add_argument("--pearson-threshold", type=float, default=-0.5)
        sp.add_argument("--batch-size", type=int, default=16)
        sp.add_argument("--max-frames", type=int, default=8)
        sp.add_argument("--force", action="store_true")

    smoke = sub.add_parser("smoke")
    common(smoke)
    smoke.set_defaults(dataset_path=SMOKE_DATASET, selection_manifest=SMOKE_MANIFEST, output_dir=ROOT / "eval_smoke", per_class=2, top_k=2, checkpoint=SMOKE_CKPT)

    single = sub.add_parser("single")
    common(single)
    single.add_argument("--checkpoint", type=Path, required=True)

    sweep = sub.add_parser("sweep")
    common(sweep)
    sweep.add_argument("--checkpoint-dir", type=Path, default=PILOT_CKPT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "sweep":
        checkpoints = discover_checkpoints(args.checkpoint_dir)
    else:
        checkpoints = [args.checkpoint]
    if not checkpoints:
        raise SystemExit("No checkpoints found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"dataset: {args.dataset_path}")
    rows = load_rows(args.dataset_path, args.per_class)
    print(f"rows: {len(rows)}")
    write_run_config(args, checkpoints)

    all_metrics = []
    summary = []
    for checkpoint in checkpoints:
        metrics = run_checkpoint(args, checkpoint, rows)
        all_metrics.extend(metrics)
        progress = [m for m in metrics if m["family"] == "progress" and m["scope"] == "sample_micro"]
        failure = [m for m in metrics if m["family"] == "failure_detection" and m["scope"] == "sample_micro"]
        summary.append({
            "step": step_from_checkpoint(checkpoint),
            "model_path": str(checkpoint),
            "progress_sample_micro_mae": progress[0]["mae"] if progress else None,
            "progress_sample_micro_spearman": progress[0]["spearman"] if progress else None,
            "failure_sample_micro_balanced_accuracy": failure[0]["balanced_accuracy"] if failure else None,
            "failure_sample_micro_auroc": failure[0]["auroc"] if failure else None,
        })
        atomic_json(args.output_dir / "metrics.json", all_metrics)
        atomic_json(args.output_dir / "sweep_summary.json", summary)

    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
