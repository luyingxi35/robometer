#!/usr/bin/env python3
"""
Run dual-model per-class reward-alignment evaluation on local/PegInsertionVertical_eval.

This reuses the baseline model/evaluation pipeline from run_baseline_eval.py, but
splits the local evaluation dataset into normalized quality groups and evaluates
both a baseline checkpoint and an oracle checkpoint on the exact same sampled
reward-alignment examples.
"""

import copy
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from hydra import main as hydra_main
from omegaconf import DictConfig

from robometer.configs.eval_configs import BaselineEvalConfig
from robometer.configs.experiment_configs import DataConfig
from robometer.data.dataset_types import ProgressSample
from robometer.data.datasets.helpers import load_frames_from_npz
from robometer.evals.baselines.gvl import GVL
from robometer.evals.baselines.rbm_model import RBMModel
from robometer.evals.baselines.robodopamine import RoboDopamine
from robometer.evals.baselines.roboreward import RoboReward
from robometer.evals.baselines.topreward import TopReward
from robometer.evals.baselines.vlac import VLAC
from robometer.evals.compile_results import run_reward_alignment_eval_per_trajectory
from robometer.evals.eval_metrics_utils import compute_spearman
from robometer.evals.run_baseline_eval import (
    _create_plot_with_video_gif,
    _make_json_serializable,
    _normalize_model_path,
    _shorten_dataset_name,
    process_batched_rbm_samples,
    process_progress_sample,
)
from robometer.utils.config_utils import convert_hydra_to_dataclass, display_config
from robometer.utils.distributed import is_rank_0
from robometer.utils.logger import get_logger

logger = get_logger()

DEFAULT_EVAL_DATASET = "local_PegInsertionVertical_eval"
DEFAULT_REPORT_FILENAME = "peg_insertion_vertical_eval_report.json"
COMPARISON_REPORT_FILENAME = "peg_insertion_vertical_comparison_report.json"
DEFAULT_BASELINE_MODEL_PATH = "robometer/Robometer-4B"
MODEL_SCOPES = ("baseline", "oracle")
QUALITY_ALIASES: Dict[str, Set[str]] = {
    "success": {"successful", "successful_labeled", "optimal"},
    "suboptimal": {"suboptimal", "suboptimal_labeled"},
    "failure": {"failure", "failed", "failure_labeled"},
}
MODEL_COLORS = {
    "gt": "#1f77b4",
    "baseline": "#ff7f0e",
    "oracle": "#2ca02c",
}


def _ensure_required_combined_indices(combined_indices: Dict[str, Any]) -> Dict[str, Any]:
    normalized_indices = copy.deepcopy(combined_indices)
    normalized_indices.setdefault("paired_human_robot_by_task", {})

    if "tasks_with_multiple_quality_labels" not in normalized_indices:
        optimal_tasks = {task for task, indices in normalized_indices.get("optimal_by_task", {}).items() if indices}
        suboptimal_tasks = {task for task, indices in normalized_indices.get("suboptimal_by_task", {}).items() if indices}
        normalized_indices["tasks_with_multiple_quality_labels"] = sorted(optimal_tasks & suboptimal_tasks)

    return normalized_indices


def _load_local_processed_dataset(dataset_name: str) -> Tuple[Any, Dict[str, Any], str]:
    cache_dir = os.environ.get("ROBOMETER_PROCESSED_DATASETS_PATH", "")
    if not cache_dir:
        raise ValueError(
            "ROBOMETER_PROCESSED_DATASETS_PATH not set. Set it to the directory containing your processed datasets."
        )

    dataset_dir = Path(cache_dir) / dataset_name
    processed_dir = dataset_dir / "processed_dataset"
    index_file = dataset_dir / "index_mappings.json"

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {processed_dir}")
    if not index_file.exists():
        raise FileNotFoundError(f"Index mappings not found: {index_file}")

    dataset = load_from_disk(str(processed_dir))
    with open(index_file) as f:
        combined_indices = json.load(f)

    combined_indices = _ensure_required_combined_indices(combined_indices)
    return dataset, combined_indices, str(dataset_dir)


def _normalize_quality_label(raw_label: Optional[str]) -> Optional[str]:
    if raw_label is None:
        return None
    raw = str(raw_label)
    for normalized_label, aliases in QUALITY_ALIASES.items():
        if raw in aliases:
            return normalized_label
    return None


def _compute_quality_counts(dataset: Any) -> Tuple[Dict[str, int], Dict[str, int], List[str]]:
    raw_counts = Counter(str(label) if label is not None else "null" for label in dataset["quality_label"])
    normalized_counts = {label: 0 for label in QUALITY_ALIASES}
    unmapped = set()

    for raw_label, count in raw_counts.items():
        normalized = _normalize_quality_label(raw_label)
        if normalized is None:
            unmapped.add(raw_label)
            continue
        normalized_counts[normalized] += count

    return dict(raw_counts), normalized_counts, sorted(unmapped)


def _filter_combined_indices(combined_indices: Dict[str, Any], keep_indices: List[int]) -> Dict[str, Any]:
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    filtered_combined_indices = {}

    for key, value in combined_indices.items():
        if isinstance(value, list):
            if value and all(isinstance(item, (int, np.integer)) for item in value):
                filtered_combined_indices[key] = [old_to_new[idx] for idx in value if idx in old_to_new]
            else:
                filtered_combined_indices[key] = [item for item in value if item in old_to_new]
        elif isinstance(value, dict):
            filtered_dict = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    filtered_nested_dict = {}
                    for nested_key, nested_list in subvalue.items():
                        if isinstance(nested_list, list):
                            remapped = [old_to_new[idx] for idx in nested_list if idx in old_to_new]
                            if remapped:
                                filtered_nested_dict[nested_key] = remapped
                    if filtered_nested_dict:
                        filtered_dict[subkey] = filtered_nested_dict
                elif isinstance(subvalue, list):
                    remapped = [old_to_new[idx] for idx in subvalue if idx in old_to_new]
                    if remapped:
                        filtered_dict[subkey] = remapped
                else:
                    filtered_dict[subkey] = subvalue
            filtered_combined_indices[key] = filtered_dict
        else:
            filtered_combined_indices[key] = value

    return _ensure_required_combined_indices(filtered_combined_indices)


def _build_reward_alignment_sampler(
    cfg: BaselineEvalConfig,
    base_data_cfg: DataConfig,
    dataset: Any,
    combined_indices: Dict[str, Any],
) -> Any:
    from robometer.data.samplers.eval.reward_alignment import RewardAlignmentSampler

    eval_data_cfg = copy.deepcopy(base_data_cfg)
    eval_data_cfg.dataset_type = "rbm"
    eval_data_cfg.eval_datasets = [DEFAULT_EVAL_DATASET]

    sampler_kwargs = {
        "config": eval_data_cfg,
        "dataset": dataset,
        "combined_indices": combined_indices,
        "dataset_success_cutoff_map": {},
        "verbose": True,
        "random_seed": cfg.custom_eval.custom_eval_random_seed,
        "max_trajectories": cfg.custom_eval.reward_alignment_max_trajectories,
        "use_frame_steps": cfg.custom_eval.use_frame_steps,
        "subsample_n_frames": cfg.custom_eval.subsample_n_frames,
        "pad_frames": cfg.custom_eval.pad_frames,
    }
    return RewardAlignmentSampler(**sampler_kwargs)


def _build_sampler_materialized_samples(sampler: Any) -> Tuple[List[Any], List[Tuple[str, int]]]:
    samples = [sampler[i] for i in range(len(sampler))]
    sample_keys: List[Tuple[str, int]] = []
    for sample in samples:
        traj = sample.trajectory
        trajectory_id = str(traj.id)
        frame_step = int(traj.metadata.get("frame_step", -1)) if traj.metadata else -1
        sample_keys.append((trajectory_id, frame_step))
    return samples, sample_keys


def _initialize_model_for_path(cfg: BaselineEvalConfig, model_path: str):
    model_config_dict = (
        asdict(cfg.model_config) if hasattr(cfg.model_config, "__dataclass_fields__") else cfg.model_config.__dict__
    )

    if cfg.reward_model == "gvl":
        return GVL(max_frames=cfg.max_frames, **model_config_dict)
    if cfg.reward_model == "vlac":
        return VLAC(model_path=model_path, **model_config_dict)
    if cfg.reward_model == "robodopamine":
        return RoboDopamine(model_path=model_path, **model_config_dict)
    if cfg.reward_model == "topreward":
        resolved_path = model_path or "Qwen/Qwen3-VL-8B-Instruct"
        return TopReward(model_path=resolved_path, **model_config_dict)
    if cfg.reward_model == "roboreward":
        resolved_path = model_path or "teetone/RoboReward-4B"
        return RoboReward(model_path=resolved_path, **model_config_dict)
    if cfg.reward_model in ["rewind", "rbm"]:
        return RBMModel(checkpoint_path=model_path)

    raise ValueError(
        f"Unsupported reward_model for progress evaluation: {cfg.reward_model}. "
        "Must be one of: gvl, vlac, robodopamine, topreward, roboreward, rbm, rewind."
    )


def _initialize_models(cfg: BaselineEvalConfig) -> Dict[str, Any]:
    oracle_model_path = cfg.model_path
    if not oracle_model_path:
        raise ValueError("model_path is required and is treated as the oracle path for dual-model local PegInsertionVertical evaluation")

    model_paths = {
        "baseline": DEFAULT_BASELINE_MODEL_PATH,
        "oracle": oracle_model_path,
    }
    models = {}
    for scope, model_path in model_paths.items():
        logger.info(f"Loading {scope} model from: {model_path}")
        models[scope] = _initialize_model_for_path(cfg, model_path)
    return models


def _normalize_eval_results_quality_labels(
    eval_results: List[Dict[str, Any]],
    normalized_label: str,
    model_scope: str,
    model_path: str,
) -> List[Dict[str, Any]]:
    normalized_results: List[Dict[str, Any]] = []
    for result in eval_results:
        updated_result = copy.deepcopy(result)
        raw_quality_label = updated_result.get("quality_label")
        updated_result["raw_quality_label"] = raw_quality_label
        updated_result["quality_label"] = normalized_label
        updated_result["model_scope"] = model_scope
        updated_result["model_path"] = model_path
        normalized_results.append(updated_result)
    return normalized_results


def _compute_per_class_reward_alignment_metrics(eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped_results: Dict[str, List[Dict[str, Any]]] = {}
    for result in eval_results:
        trajectory_id = result.get("id")
        if not trajectory_id:
            continue
        grouped_results.setdefault(trajectory_id, []).append(result)

    mse_values: List[float] = []
    spearman_values: List[float] = []

    for trajectory_results in grouped_results.values():
        trajectory_results.sort(key=lambda r: r.get("metadata", {}).get("frame_step", 0))
        traj_preds = np.array([row["progress_pred"][-1] for row in trajectory_results], dtype=float)
        traj_targets = np.array([row["target_progress"][-1] for row in trajectory_results], dtype=float)

        if traj_preds.size == 0 or traj_targets.size == 0 or traj_preds.size != traj_targets.size:
            continue

        mse_values.append(float(np.mean((traj_targets - traj_preds) ** 2)))

        spearman_value = compute_spearman(traj_targets.tolist(), traj_preds.tolist())
        if not np.isnan(spearman_value):
            spearman_values.append(float(spearman_value))

    return {
        "num_trajectories": len(grouped_results),
        "num_results": len(eval_results),
        "num_valid_spearman_trajectories": len(spearman_values),
        "mse": float(np.mean(mse_values)) if mse_values else None,
        "spearman": float(np.mean(spearman_values)) if spearman_values else None,
    }


def _process_reward_alignment_samples(
    cfg: BaselineEvalConfig,
    samples: List[Any],
    model: Any,
    model_scope: str,
    model_path: str,
    normalized_label: str,
) -> List[Dict[str, Any]]:
    eval_results: List[Dict[str, Any]] = []

    if cfg.reward_model in ["rewind", "rbm"]:
        model_config_dict = (
            asdict(cfg.model_config) if hasattr(cfg.model_config, "__dataclass_fields__") else cfg.model_config.__dict__
        )
        eval_results.extend(
            process_batched_rbm_samples(samples, model, batch_size=model_config_dict["batch_size"])
        )
    else:
        for sample in samples:
            if not isinstance(sample, ProgressSample):
                logger.warning(f"Sample type mismatch for reward_alignment: {type(sample)}")
                continue
            result = process_progress_sample(sample, model)
            if result:
                eval_results.append(result)

    return _normalize_eval_results_quality_labels(eval_results, normalized_label, model_scope, model_path)


def _validate_aligned_results(
    sample_keys: List[Tuple[str, int]],
    baseline_results: List[Dict[str, Any]],
    oracle_results: List[Dict[str, Any]],
    normalized_label: str,
) -> None:
    def build_keys(results: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        keys: List[Tuple[str, int]] = []
        for result in results:
            metadata = result.get("metadata", {}) or {}
            keys.append((str(result.get("id")), int(metadata.get("frame_step", -1))))
        return keys

    baseline_keys = build_keys(baseline_results)
    oracle_keys = build_keys(oracle_results)

    if len(sample_keys) != len(baseline_keys) or len(sample_keys) != len(oracle_keys):
        raise ValueError(
            f"Result count mismatch for class {normalized_label}: "
            f"samples={len(sample_keys)}, baseline={len(baseline_keys)}, oracle={len(oracle_keys)}"
        )

    if baseline_keys != sample_keys:
        raise ValueError(f"Baseline results are not aligned with sampled keys for class {normalized_label}")
    if oracle_keys != sample_keys:
        raise ValueError(f"Oracle results are not aligned with sampled keys for class {normalized_label}")


def _save_model_outputs(
    class_output_dir: str,
    short_dataset_name: str,
    eval_results: List[Dict[str, Any]],
    metrics_dict: Dict[str, Any],
    plots: List[Any],
    video_frames_list: List[Any],
) -> None:
    os.makedirs(class_output_dir, exist_ok=True)

    results_file = os.path.join(class_output_dir, f"{short_dataset_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(_make_json_serializable(eval_results), f, indent=2)

    metrics_file = os.path.join(class_output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(_make_json_serializable(metrics_dict), f, indent=2)

    if plots:
        plots_dir = os.path.join(class_output_dir, f"{short_dataset_name}_plots")
        os.makedirs(plots_dir, exist_ok=True)
        for i, fig in enumerate(plots[:10]):
            video_frames = video_frames_list[i] if i < len(video_frames_list) else None
            gif_path = os.path.join(plots_dir, f"trajectory_{i:04d}.gif")
            _create_plot_with_video_gif(fig, video_frames, gif_path)


def _build_output_dir(cfg: BaselineEvalConfig) -> str:
    if cfg.output_dir is not None:
        return cfg.output_dir

    normalized_path = _normalize_model_path(cfg.model_path)
    if normalized_path:
        dir_name = f"{cfg.reward_model}_{normalized_path}_local_peg_insertion_vertical"
    else:
        dir_name = f"{cfg.reward_model}_local_peg_insertion_vertical"
    return os.path.join("./baseline_eval_output", dir_name)


def _plot_metric_comparison_bars(comparison_dir: str, comparison_report: Dict[str, Any]) -> None:
    metrics_dir = os.path.join(comparison_dir, "metrics_plots")
    os.makedirs(metrics_dir, exist_ok=True)

    for normalized_label, class_payload in comparison_report["per_class"].items():
        baseline_metrics = class_payload["baseline"]["metrics"]
        oracle_metrics = class_payload["oracle"]["metrics"]
        metric_names = ["mse", "spearman"]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"{normalized_label} metrics comparison", fontsize=12)
        for ax, metric_name in zip(axes, metric_names):
            baseline_value = baseline_metrics.get(metric_name)
            oracle_value = oracle_metrics.get(metric_name)
            values = [baseline_value if baseline_value is not None else 0.0, oracle_value if oracle_value is not None else 0.0]
            colors = [MODEL_COLORS["baseline"], MODEL_COLORS["oracle"]]
            labels = ["baseline", "oracle"]
            bars = ax.bar(labels, values, color=colors)
            ax.set_title(metric_name.upper())
            ax.set_ylabel(metric_name)
            for bar, value in zip(bars, [baseline_value, oracle_value]):
                label = "None" if value is None else f"{value:.4f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(metrics_dir, f"{normalized_label}_metrics_comparison.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


def _group_results_by_trajectory(eval_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in eval_results:
        trajectory_id = str(result.get("id"))
        grouped_results[trajectory_id].append(result)
    for trajectory_id in grouped_results:
        grouped_results[trajectory_id].sort(key=lambda r: r.get("metadata", {}).get("frame_step", 0))
    return dict(grouped_results)


def _plot_trajectory_comparison(
    output_path: str,
    trajectory_id: str,
    quality_label: str,
    baseline_rows: List[Dict[str, Any]],
    oracle_rows: List[Dict[str, Any]],
) -> None:
    baseline_frame_steps = [int(row.get("metadata", {}).get("frame_step", -1)) for row in baseline_rows]
    oracle_frame_steps = [int(row.get("metadata", {}).get("frame_step", -1)) for row in oracle_rows]
    if baseline_frame_steps != oracle_frame_steps:
        raise ValueError(f"Frame-step mismatch for trajectory {trajectory_id}")

    video_path = baseline_rows[0].get("video_path")
    frames = load_frames_from_npz(video_path)
    selected_frames = [frames[min(max(step, 0), len(frames) - 1)] for step in baseline_frame_steps]

    gt_curve = [float(row["target_progress"][-1]) for row in baseline_rows]
    baseline_curve = [float(row["progress_pred"][-1]) for row in baseline_rows]
    oracle_curve = [float(row["progress_pred"][-1]) for row in oracle_rows]

    ncols = len(selected_frames)
    fig_width = max(12.0, ncols * 1.8)
    fig = plt.figure(figsize=(fig_width, 8.5))
    grid = fig.add_gridspec(2, ncols, height_ratios=[1.0, 1.6], hspace=0.08, wspace=0.04)

    for col, (frame, frame_step) in enumerate(zip(selected_frames, baseline_frame_steps)):
        ax = fig.add_subplot(grid[0, col])
        if frame.ndim == 3 and frame.shape[0] in [1, 3]:
            frame = np.transpose(frame, (1, 2, 0))
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        ax.imshow(frame)
        ax.set_title(f"step {frame_step}", fontsize=7)
        ax.axis("off")

    ax_curve = fig.add_subplot(grid[1, :])
    x = np.arange(len(baseline_frame_steps))
    ax_curve.plot(x, gt_curve, marker="o", linewidth=2.0, color=MODEL_COLORS["gt"], label="ground truth")
    ax_curve.plot(x, baseline_curve, marker="s", linewidth=2.0, color=MODEL_COLORS["baseline"], label="baseline")
    ax_curve.plot(x, oracle_curve, marker="^", linewidth=2.0, color=MODEL_COLORS["oracle"], label="oracle")
    ax_curve.set_xticks(x)
    ax_curve.set_xticklabels([str(step) for step in baseline_frame_steps], rotation=45, ha="right")
    ax_curve.set_xlabel("sampled prefix endpoint (frame_step)")
    ax_curve.set_ylabel("progress")
    ax_curve.set_ylim(-0.03, 1.03)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="best", fontsize=8)
    ax_curve.set_title(f"{trajectory_id} | {quality_label}", fontsize=10)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _create_trajectory_comparison_visualizations(comparison_dir: str, comparison_report: Dict[str, Any]) -> None:
    trajectory_root = os.path.join(comparison_dir, "trajectory_plots")
    os.makedirs(trajectory_root, exist_ok=True)

    for normalized_label, class_payload in comparison_report["per_class"].items():
        baseline_rows = class_payload["baseline"]["results"]
        oracle_rows = class_payload["oracle"]["results"]
        baseline_grouped = _group_results_by_trajectory(baseline_rows)
        oracle_grouped = _group_results_by_trajectory(oracle_rows)

        if set(baseline_grouped.keys()) != set(oracle_grouped.keys()):
            raise ValueError(f"Trajectory sets do not match for class {normalized_label}")

        class_dir = os.path.join(trajectory_root, normalized_label)
        os.makedirs(class_dir, exist_ok=True)
        for trajectory_id in sorted(baseline_grouped.keys()):
            output_path = os.path.join(class_dir, f"{trajectory_id}.png")
            _plot_trajectory_comparison(
                output_path=output_path,
                trajectory_id=trajectory_id,
                quality_label=normalized_label,
                baseline_rows=baseline_grouped[trajectory_id],
                oracle_rows=oracle_grouped[trajectory_id],
            )


def _run_dual_model_reward_alignment(
    cfg: BaselineEvalConfig,
    base_data_cfg: DataConfig,
    dataset: Any,
    combined_indices: Dict[str, Any],
    models: Dict[str, Any],
    model_paths: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    short_dataset_name = _shorten_dataset_name(DEFAULT_EVAL_DATASET)
    per_model_reports: Dict[str, Any] = {scope: {} for scope in MODEL_SCOPES}
    comparison_report: Dict[str, Any] = {"per_class": {}}

    for normalized_label, aliases in QUALITY_ALIASES.items():
        class_keep_indices = [
            idx for idx, raw_label in enumerate(dataset["quality_label"]) if str(raw_label) in aliases
        ]
        class_payload = {
            "aliases": sorted(aliases),
            "raw_count": len(class_keep_indices),
        }

        if not class_keep_indices:
            for scope in MODEL_SCOPES:
                per_model_reports[scope][normalized_label] = {
                    "aliases": sorted(aliases),
                    "raw_count": 0,
                    "evaluated_trajectory_count": 0,
                    "status": "skipped",
                    "metrics": {},
                    "results_count": 0,
                }
            comparison_report["per_class"][normalized_label] = {
                **class_payload,
                "baseline": {"metrics": {}, "results": []},
                "oracle": {"metrics": {}, "results": []},
                "deltas": {},
            }
            continue

        filtered_dataset = dataset.select(class_keep_indices)
        filtered_indices = _filter_combined_indices(combined_indices, class_keep_indices)
        sampler = _build_reward_alignment_sampler(cfg, base_data_cfg, filtered_dataset, filtered_indices)
        materialized_samples, sample_keys = _build_sampler_materialized_samples(sampler)

        model_results: Dict[str, List[Dict[str, Any]]] = {}
        model_metrics: Dict[str, Dict[str, Any]] = {}

        for scope in MODEL_SCOPES:
            eval_results = _process_reward_alignment_samples(
                cfg=cfg,
                samples=materialized_samples,
                model=models[scope],
                model_scope=scope,
                model_path=model_paths[scope],
                normalized_label=normalized_label,
            )
            model_results[scope] = eval_results
            model_metrics[scope] = _compute_per_class_reward_alignment_metrics(eval_results)

            data_source = eval_results[0].get("data_source") if eval_results else None
            _, plots, video_frames_list, _ = run_reward_alignment_eval_per_trajectory(
                results=eval_results,
                progress_pred_type="absolute_wrt_total_frames",
                is_discrete_mode=False,
                num_bins=None,
                data_source=data_source,
                use_frame_steps=cfg.custom_eval.use_frame_steps,
                train_success_head=False,
                last_frame_only=False,
            ) if eval_results else ({}, [], [], [])

            class_output_dir = os.path.join(cfg.output_dir, scope, "reward_alignment", normalized_label)
            _save_model_outputs(
                class_output_dir=class_output_dir,
                short_dataset_name=short_dataset_name,
                eval_results=eval_results,
                metrics_dict=model_metrics[scope],
                plots=plots,
                video_frames_list=video_frames_list,
            )

            per_model_reports[scope][normalized_label] = {
                "aliases": sorted(aliases),
                "raw_count": len(class_keep_indices),
                "evaluated_trajectory_count": min(
                    len(filtered_indices.get("robot_trajectories", [])),
                    cfg.custom_eval.reward_alignment_max_trajectories
                    if cfg.custom_eval.reward_alignment_max_trajectories is not None
                    else len(filtered_indices.get("robot_trajectories", [])),
                ),
                "status": "completed",
                "metrics": {
                    key: float(value) if isinstance(value, (int, float, np.number)) else value
                    for key, value in model_metrics[scope].items()
                },
                "results_count": len(eval_results),
            }

        _validate_aligned_results(sample_keys, model_results["baseline"], model_results["oracle"], normalized_label)

        baseline_metrics = model_metrics["baseline"]
        oracle_metrics = model_metrics["oracle"]
        deltas = {
            "delta_spearman_oracle_minus_baseline": (
                None
                if baseline_metrics.get("spearman") is None or oracle_metrics.get("spearman") is None
                else float(oracle_metrics["spearman"] - baseline_metrics["spearman"])
            ),
            "delta_mse_baseline_minus_oracle": (
                None
                if baseline_metrics.get("mse") is None or oracle_metrics.get("mse") is None
                else float(baseline_metrics["mse"] - oracle_metrics["mse"])
            ),
        }

        comparison_report["per_class"][normalized_label] = {
            **class_payload,
            "baseline": {
                "metrics": {
                    key: float(value) if isinstance(value, (int, float, np.number)) else value
                    for key, value in baseline_metrics.items()
                },
                "results": model_results["baseline"],
            },
            "oracle": {
                "metrics": {
                    key: float(value) if isinstance(value, (int, float, np.number)) else value
                    for key, value in oracle_metrics.items()
                },
                "results": model_results["oracle"],
            },
            "deltas": deltas,
        }

    return per_model_reports, comparison_report


def _write_model_report(
    cfg: BaselineEvalConfig,
    report_path: str,
    raw_counts: Dict[str, int],
    normalized_counts: Dict[str, int],
    unmapped_labels: List[str],
    per_class_report: Dict[str, Any],
    model_scope: str,
    model_path: str,
) -> None:
    config_snapshot = {
        "reward_model": cfg.reward_model,
        "model_scope": model_scope,
        "model_path": model_path,
        "baseline_default_model_path": DEFAULT_BASELINE_MODEL_PATH,
        "max_frames": cfg.max_frames,
        "model_config": asdict(cfg.model_config) if hasattr(cfg.model_config, "__dataclass_fields__") else cfg.model_config,
        "custom_eval": asdict(cfg.custom_eval) if hasattr(cfg.custom_eval, "__dataclass_fields__") else cfg.custom_eval,
    }

    report = {
        "dataset_name": DEFAULT_EVAL_DATASET,
        "run_timestamp": datetime.now().isoformat(),
        "config": _make_json_serializable(config_snapshot),
        "raw_quality_label_counts": raw_counts,
        "normalized_quality_label_counts": normalized_counts,
        "total_trajectories": int(sum(raw_counts.values())),
        "unmapped_raw_labels": unmapped_labels,
        "per_class": per_class_report,
    }

    with open(report_path, "w") as f:
        json.dump(_make_json_serializable(report), f, indent=2)


def _write_comparison_report(
    cfg: BaselineEvalConfig,
    report_path: str,
    raw_counts: Dict[str, int],
    normalized_counts: Dict[str, int],
    unmapped_labels: List[str],
    comparison_report: Dict[str, Any],
    model_paths: Dict[str, str],
) -> None:
    config_snapshot = {
        "reward_model": cfg.reward_model,
        "baseline_model_path": model_paths["baseline"],
        "oracle_model_path": model_paths["oracle"],
        "max_frames": cfg.max_frames,
        "model_config": asdict(cfg.model_config) if hasattr(cfg.model_config, "__dataclass_fields__") else cfg.model_config,
        "custom_eval": asdict(cfg.custom_eval) if hasattr(cfg.custom_eval, "__dataclass_fields__") else cfg.custom_eval,
        "delta_definition": {
            "spearman": "oracle - baseline",
            "mse": "baseline - oracle",
        },
    }

    payload = {
        "dataset_name": DEFAULT_EVAL_DATASET,
        "run_timestamp": datetime.now().isoformat(),
        "config": _make_json_serializable(config_snapshot),
        "raw_quality_label_counts": raw_counts,
        "normalized_quality_label_counts": normalized_counts,
        "total_trajectories": int(sum(raw_counts.values())),
        "unmapped_raw_labels": unmapped_labels,
        **comparison_report,
    }

    with open(report_path, "w") as f:
        json.dump(_make_json_serializable(payload), f, indent=2)


def _strip_results_from_comparison_report(comparison_report: Dict[str, Any]) -> Dict[str, Any]:
    stripped = copy.deepcopy(comparison_report)
    for class_payload in stripped.get("per_class", {}).values():
        if "baseline" in class_payload:
            class_payload["baseline"].pop("results", None)
        if "oracle" in class_payload:
            class_payload["oracle"].pop("results", None)
    return stripped


@hydra_main(version_base=None, config_path="../configs", config_name="baseline_eval_config")
def main(cfg: DictConfig):
    baseline_cfg = convert_hydra_to_dataclass(cfg, BaselineEvalConfig)
    display_config(baseline_cfg)

    if baseline_cfg.reward_model not in ["gvl", "vlac", "roboreward", "robodopamine", "topreward", "rbm", "rewind"]:
        raise ValueError(
            "reward_model must be one of 'gvl', 'vlac', 'roboreward', 'robodopamine', 'topreward', 'rbm', or 'rewind'"
        )

    baseline_cfg.custom_eval.eval_types = ["reward_alignment"]
    baseline_cfg.custom_eval.reward_alignment = [DEFAULT_EVAL_DATASET]
    baseline_cfg.output_dir = _build_output_dir(baseline_cfg)
    os.makedirs(baseline_cfg.output_dir, exist_ok=True)
    logger.info(f"Output directory: {baseline_cfg.output_dir}")

    data_cfg = DataConfig(
        max_frames=baseline_cfg.max_frames,
        load_embeddings=True if "rewind" in baseline_cfg.reward_model else False,
    )
    display_config(data_cfg)

    dataset, combined_indices, _ = _load_local_processed_dataset(DEFAULT_EVAL_DATASET)
    raw_counts, normalized_counts, unmapped_labels = _compute_quality_counts(dataset)
    logger.info(f"Raw quality label counts: {raw_counts}")
    logger.info(f"Normalized quality label counts: {normalized_counts}")
    if unmapped_labels:
        logger.warning(f"Found unmapped quality labels: {unmapped_labels}")

    model_paths = {
        "baseline": DEFAULT_BASELINE_MODEL_PATH,
        "oracle": baseline_cfg.model_path,
    }
    models = _initialize_models(baseline_cfg)
    per_model_reports, comparison_report = _run_dual_model_reward_alignment(
        cfg=baseline_cfg,
        base_data_cfg=data_cfg,
        dataset=dataset,
        combined_indices=combined_indices,
        models=models,
        model_paths=model_paths,
    )

    if is_rank_0():
        for scope in MODEL_SCOPES:
            scope_dir = os.path.join(baseline_cfg.output_dir, scope)
            os.makedirs(scope_dir, exist_ok=True)
            report_path = os.path.join(scope_dir, DEFAULT_REPORT_FILENAME)
            _write_model_report(
                cfg=baseline_cfg,
                report_path=report_path,
                raw_counts=raw_counts,
                normalized_counts=normalized_counts,
                unmapped_labels=unmapped_labels,
                per_class_report=per_model_reports[scope],
                model_scope=scope,
                model_path=model_paths[scope],
            )
            logger.info(f"Saved {scope} report to: {report_path}")

        comparison_dir = os.path.join(baseline_cfg.output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        comparison_report_path = os.path.join(comparison_dir, COMPARISON_REPORT_FILENAME)
        _write_comparison_report(
            cfg=baseline_cfg,
            report_path=comparison_report_path,
            raw_counts=raw_counts,
            normalized_counts=normalized_counts,
            unmapped_labels=unmapped_labels,
            comparison_report=_strip_results_from_comparison_report(comparison_report),
            model_paths=model_paths,
        )
        logger.info(f"Saved comparison report to: {comparison_report_path}")

        _plot_metric_comparison_bars(comparison_dir, comparison_report)
        _create_trajectory_comparison_visualizations(comparison_dir, comparison_report)
        logger.info(f"Saved comparison visualizations to: {comparison_dir}")

    logger.info("\nDual-model PegInsertionVertical evaluation complete!")
    return {
        "baseline": per_model_reports["baseline"],
        "oracle": per_model_reports["oracle"],
        "comparison": _strip_results_from_comparison_report(comparison_report),
    }


if __name__ == "__main__":
    main()
