#!/usr/bin/env python3
"""
Run per-class reward-alignment evaluation on local/PegInsertionVertical_eval.

This reuses the baseline model/evaluation pipeline from run_baseline_eval.py, but
splits the local evaluation dataset into normalized quality groups:
success, failure, and suboptimal.
"""

import copy
import json
import os
from collections import Counter
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
from robometer.evals.compile_results import run_reward_alignment_eval_per_trajectory
from robometer.evals.run_baseline_eval import (
    _create_plot_with_video_gif,
    _make_json_serializable,
    _normalize_model_path,
    _shorten_dataset_name,
    process_batched_rbm_samples,
    process_progress_sample,
)
from robometer.evals.baselines.gvl import GVL
from robometer.evals.baselines.rbm_model import RBMModel
from robometer.evals.baselines.robodopamine import RoboDopamine
from robometer.evals.baselines.roboreward import RoboReward
from robometer.evals.baselines.topreward import TopReward
from robometer.evals.baselines.vlac import VLAC
from robometer.utils.config_utils import convert_hydra_to_dataclass, display_config
from robometer.evals.eval_metrics_utils import compute_pearson
from robometer.utils.distributed import is_rank_0
from robometer.utils.logger import get_logger
from robometer.data.dataset_types import ProgressSample

logger = get_logger()

DEFAULT_EVAL_DATASET = "local_PegInsertionVertical_eval"
DEFAULT_REPORT_FILENAME = "peg_insertion_vertical_eval_report.json"
QUALITY_ALIASES: Dict[str, Set[str]] = {
    "success": {"successful", "successful_labeled", "optimal"},
    "suboptimal": {"suboptimal", "suboptimal_labeled"},
    "failure": {"failure", "failed", "failure_labeled"},
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


def _initialize_model(cfg: BaselineEvalConfig):
    model_config_dict = (
        asdict(cfg.model_config) if hasattr(cfg.model_config, "__dataclass_fields__") else cfg.model_config.__dict__
    )

    if cfg.reward_model == "gvl":
        return GVL(max_frames=cfg.max_frames, **model_config_dict)
    if cfg.reward_model == "vlac":
        if not cfg.model_path:
            raise ValueError("model_path is required for VLAC baseline")
        return VLAC(model_path=cfg.model_path, **model_config_dict)
    if cfg.reward_model == "robodopamine":
        if not cfg.model_path:
            raise ValueError("model_path is required for Robo-Dopamine baseline")
        return RoboDopamine(model_path=cfg.model_path, **model_config_dict)
    if cfg.reward_model == "topreward":
        model_path = cfg.model_path or "Qwen/Qwen3-VL-8B-Instruct"
        return TopReward(model_path=model_path, **model_config_dict)
    if cfg.reward_model == "roboreward":
        return RoboReward(model_path=cfg.model_path or "teetone/RoboReward-4B", **model_config_dict)
    if cfg.reward_model in ["rewind", "rbm"]:
        if not cfg.model_path:
            raise ValueError("model_path is required for RBM/ReWiND reward model")
        return RBMModel(checkpoint_path=cfg.model_path)

    raise ValueError(
        f"Unsupported reward_model for progress evaluation: {cfg.reward_model}. "
        "Must be one of: gvl, vlac, robodopamine, topreward, roboreward, rbm, rewind."
    )


def _normalize_eval_results_quality_labels(
    eval_results: List[Dict[str, Any]],
    normalized_label: str,
) -> List[Dict[str, Any]]:
    normalized_results: List[Dict[str, Any]] = []
    for result in eval_results:
        updated_result = copy.deepcopy(result)
        raw_quality_label = updated_result.get("quality_label")
        updated_result["raw_quality_label"] = raw_quality_label
        updated_result["quality_label"] = normalized_label
        normalized_results.append(updated_result)
    return normalized_results


def _compute_per_class_reward_alignment_metrics(eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped_results: Dict[str, List[Dict[str, Any]]] = {}
    for result in eval_results:
        trajectory_id = result.get("id")
        if not trajectory_id:
            continue
        grouped_results.setdefault(trajectory_id, []).append(result)

    loss_values: List[float] = []
    pearson_values: List[float] = []

    for trajectory_id, trajectory_results in grouped_results.items():
        trajectory_results.sort(key=lambda r: r.get("metadata", {}).get("frame_step", 0))
        traj_preds = np.array([row["progress_pred"][-1] for row in trajectory_results], dtype=float)
        traj_targets = np.array([row["target_progress"][-1] for row in trajectory_results], dtype=float)

        if traj_preds.size == 0 or traj_targets.size == 0 or traj_preds.size != traj_targets.size:
            continue

        loss_values.append(float(np.mean((traj_targets - traj_preds) ** 2)))

        pearson_value = compute_pearson(traj_targets.tolist(), traj_preds.tolist())
        if not np.isnan(pearson_value):
            pearson_values.append(float(pearson_value))

    metrics: Dict[str, Any] = {
        "num_trajectories": len(grouped_results),
        "num_results": len(eval_results),
        "loss": float(np.mean(loss_values)) if loss_values else None,
        "pearson": float(np.mean(pearson_values)) if pearson_values else None,
    }
    return metrics


def _process_reward_alignment_dataset(
    cfg: BaselineEvalConfig,
    sampler: Any,
    model: Any,
) -> List[Dict[str, Any]]:
    eval_results: List[Dict[str, Any]] = []

    if cfg.reward_model in ["rewind", "rbm"]:
        model_config_dict = (
            asdict(cfg.model_config) if hasattr(cfg.model_config, "__dataclass_fields__") else cfg.model_config.__dict__
        )
        eval_results.extend(
            process_batched_rbm_samples(sampler, model, batch_size=model_config_dict["batch_size"])
        )
        return eval_results

    for sample in sampler:
        if not isinstance(sample, ProgressSample):
            logger.warning(f"Sample type mismatch for reward_alignment: {type(sample)}")
            continue
        result = process_progress_sample(sample, model)
        if result:
            eval_results.append(result)

    return eval_results


def _save_class_outputs(
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


def _run_per_class_reward_alignment(
    cfg: BaselineEvalConfig,
    base_data_cfg: DataConfig,
    dataset: Any,
    combined_indices: Dict[str, Any],
    model: Any,
) -> Dict[str, Any]:
    eval_type_dir = os.path.join(cfg.output_dir, "reward_alignment")
    os.makedirs(eval_type_dir, exist_ok=True)

    short_dataset_name = _shorten_dataset_name(DEFAULT_EVAL_DATASET)
    per_class_report: Dict[str, Any] = {}

    for normalized_label, aliases in QUALITY_ALIASES.items():
        class_keep_indices = [
            idx
            for idx, raw_label in enumerate(dataset["quality_label"])
            if str(raw_label) in aliases
        ]
        class_output_dir = os.path.join(eval_type_dir, normalized_label)

        class_report = {
            "aliases": sorted(aliases),
            "raw_count": len(class_keep_indices),
            "evaluated_trajectory_count": 0,
            "status": "skipped",
            "metrics": {},
            "results_count": 0,
        }

        if not class_keep_indices:
            logger.info(f"No trajectories found for class '{normalized_label}', skipping inference.")
            per_class_report[normalized_label] = class_report
            continue

        filtered_dataset = dataset.select(class_keep_indices)
        filtered_indices = _filter_combined_indices(combined_indices, class_keep_indices)
        sampler = _build_reward_alignment_sampler(cfg, base_data_cfg, filtered_dataset, filtered_indices)

        class_report["evaluated_trajectory_count"] = min(
            len(filtered_indices.get("robot_trajectories", [])),
            cfg.custom_eval.reward_alignment_max_trajectories
            if cfg.custom_eval.reward_alignment_max_trajectories is not None
            else len(filtered_indices.get("robot_trajectories", [])),
        )

        eval_results = _process_reward_alignment_dataset(cfg, sampler, model)
        eval_results = _normalize_eval_results_quality_labels(eval_results, normalized_label)
        class_report["results_count"] = len(eval_results)

        if not eval_results:
            logger.warning(f"No reward-alignment results produced for class '{normalized_label}'.")
            per_class_report[normalized_label] = class_report
            continue

        data_source = eval_results[0].get("data_source")
        _, plots, video_frames_list, _ = run_reward_alignment_eval_per_trajectory(
            results=eval_results,
            progress_pred_type="absolute_wrt_total_frames",
            is_discrete_mode=False,
            num_bins=None,
            data_source=data_source,
            use_frame_steps=cfg.custom_eval.use_frame_steps,
            train_success_head=False,
            last_frame_only=False,
        )
        metrics_dict = _compute_per_class_reward_alignment_metrics(eval_results)

        _save_class_outputs(
            class_output_dir=class_output_dir,
            short_dataset_name=short_dataset_name,
            eval_results=eval_results,
            metrics_dict=metrics_dict,
            plots=plots,
            video_frames_list=video_frames_list,
        )

        class_report["status"] = "completed"
        class_report["metrics"] = {
            key: float(value) if isinstance(value, (int, float, np.number)) else value
            for key, value in metrics_dict.items()
        }
        per_class_report[normalized_label] = class_report

    return per_class_report


def _build_output_dir(cfg: BaselineEvalConfig) -> str:
    if cfg.output_dir is not None:
        return cfg.output_dir

    normalized_path = _normalize_model_path(cfg.model_path)
    if normalized_path:
        dir_name = f"{cfg.reward_model}_{normalized_path}_local_peg_insertion_vertical"
    else:
        dir_name = f"{cfg.reward_model}_local_peg_insertion_vertical"
    return os.path.join("./baseline_eval_output", dir_name)


def _write_final_report(
    cfg: BaselineEvalConfig,
    report_path: str,
    raw_counts: Dict[str, int],
    normalized_counts: Dict[str, int],
    unmapped_labels: List[str],
    per_class_report: Dict[str, Any],
) -> None:
    config_snapshot = {
        "reward_model": cfg.reward_model,
        "model_path": cfg.model_path,
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

    model = _initialize_model(baseline_cfg)
    per_class_report = _run_per_class_reward_alignment(
        cfg=baseline_cfg,
        base_data_cfg=data_cfg,
        dataset=dataset,
        combined_indices=combined_indices,
        model=model,
    )

    if is_rank_0():
        report_path = os.path.join(baseline_cfg.output_dir, DEFAULT_REPORT_FILENAME)
        _write_final_report(
            cfg=baseline_cfg,
            report_path=report_path,
            raw_counts=raw_counts,
            normalized_counts=normalized_counts,
            unmapped_labels=unmapped_labels,
            per_class_report=per_class_report,
        )
        logger.info(f"Saved aggregate report to: {report_path}")

    logger.info("\nPer-class PegInsertionVertical baseline evaluation complete!")
    return per_class_report


if __name__ == "__main__":
    main()
