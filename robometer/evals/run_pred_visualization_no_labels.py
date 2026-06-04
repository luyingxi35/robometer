#!/usr/bin/env python3
"""
Visualize dual-model progress predictions on a local dataset that lacks target_progress labels.

This script mirrors the structure of run_baseline_eval_local_peg_insertion_vertical.py but:
  - Does NOT require target_progress or quality_label columns in the dataset.
  - Does NOT compute any metrics (no MAE, Spearman, reward-alignment score).
  - Only visualizes predicted progress curves from the baseline checkpoint
    (robometer/Robometer-4B) and the user-supplied oracle checkpoint side by side.

Output layout:
  <output_dir>/
    baseline/results.json
    oracle/results.json
    comparison/
      trajectory_plots/<trajectory_id>.png
      run_report.json
"""

import json
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from hydra import main as hydra_main
from omegaconf import DictConfig

from robometer.configs.eval_configs import BaselineEvalConfig
from robometer.configs.experiment_configs import DataConfig
from robometer.data.datasets.helpers import load_frames_from_npz
from robometer.evals.baselines.gvl import GVL
from robometer.evals.baselines.rbm_model import RBMModel
from robometer.evals.baselines.robodopamine import RoboDopamine
from robometer.evals.baselines.roboreward import RoboReward
from robometer.evals.baselines.topreward import TopReward
from robometer.evals.baselines.vlac import VLAC
from robometer.evals.run_baseline_eval import (
    _make_json_serializable,
    _normalize_model_path,
    _shorten_dataset_name,
    process_batched_rbm_samples,
    process_progress_sample,
)
from robometer.data.dataset_types import ProgressSample
from robometer.utils.config_utils import convert_hydra_to_dataclass, display_config
from robometer.utils.distributed import is_rank_0
from robometer.utils.logger import get_logger

logger = get_logger()

DEFAULT_EVAL_DATASET = "local_real_eval"
DEFAULT_BASELINE_MODEL_PATH = "/home/yingxi/RoboFAC/robometer/robometer/Robometer-4B"
MODEL_SCOPES = ("baseline", "oracle")
MODEL_COLORS = {
    "baseline": "#ff7f0e",
    "oracle": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_local_processed_dataset(dataset_name: str) -> Tuple[Any, Dict[str, Any], str]:
    cache_dir = os.environ.get("ROBOMETER_PROCESSED_DATASETS_PATH", "")
    if not cache_dir:
        raise ValueError(
            "ROBOMETER_PROCESSED_DATASETS_PATH not set. "
            "Set it to the directory containing your processed datasets."
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

    combined_indices.setdefault("paired_human_robot_by_task", {})
    return dataset, combined_indices, str(dataset_dir)


# ---------------------------------------------------------------------------
# Model initialization (identical to original)
# ---------------------------------------------------------------------------


def _initialize_model_for_path(cfg: BaselineEvalConfig, model_path: str) -> Any:
    model_config_dict = (
        asdict(cfg.model_config)
        if hasattr(cfg.model_config, "__dataclass_fields__")
        else cfg.model_config.__dict__
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
        f"Unsupported reward_model: {cfg.reward_model}. "
        "Must be one of: gvl, vlac, robodopamine, topreward, roboreward, rbm, rewind."
    )


def _initialize_models(cfg: BaselineEvalConfig) -> Dict[str, Any]:
    oracle_model_path = cfg.model_path
    if not oracle_model_path:
        raise ValueError(
            "model_path is required (treated as the oracle checkpoint path)"
        )

    model_paths = {
        "baseline": DEFAULT_BASELINE_MODEL_PATH,
        "oracle": oracle_model_path,
    }
    models = {}
    for scope, path in model_paths.items():
        logger.info(f"Loading {scope} model from: {path}")
        models[scope] = _initialize_model_for_path(cfg, path)
    return models


# ---------------------------------------------------------------------------
# Sampler helpers (identical to original)
# ---------------------------------------------------------------------------


def _build_reward_alignment_sampler(
    cfg: BaselineEvalConfig,
    base_data_cfg: DataConfig,
    dataset: Any,
    combined_indices: Dict[str, Any],
) -> Any:
    import copy
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


# ---------------------------------------------------------------------------
# Inference (identical to original)
# ---------------------------------------------------------------------------


def _process_samples(
    cfg: BaselineEvalConfig,
    samples: List[Any],
    model: Any,
    model_scope: str,
    model_path: str,
) -> List[Dict[str, Any]]:
    eval_results: List[Dict[str, Any]] = []

    if cfg.reward_model in ["rewind", "rbm"]:
        model_config_dict = (
            asdict(cfg.model_config)
            if hasattr(cfg.model_config, "__dataclass_fields__")
            else cfg.model_config.__dict__
        )
        eval_results.extend(
            process_batched_rbm_samples(samples, model, batch_size=model_config_dict["batch_size"])
        )
    else:
        for sample in samples:
            if not isinstance(sample, ProgressSample):
                logger.warning(f"Sample type mismatch: {type(sample)}")
                continue
            result = process_progress_sample(sample, model)
            if result:
                eval_results.append(result)

    # Tag each result with scope / path (target_progress is intentionally ignored)
    for result in eval_results:
        result["model_scope"] = model_scope
        result["model_path"] = model_path

    return eval_results


# ---------------------------------------------------------------------------
# Grouping helper (identical to original)
# ---------------------------------------------------------------------------


def _group_results_by_trajectory(eval_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in eval_results:
        trajectory_id = str(result.get("id"))
        grouped[trajectory_id].append(result)
    for trajectory_id in grouped:
        grouped[trajectory_id].sort(
            key=lambda r: r.get("metadata", {}).get("frame_step", 0)
        )
    return dict(grouped)


# ---------------------------------------------------------------------------
# Visualization — predictions only, no GT curve
# ---------------------------------------------------------------------------


def _plot_trajectory_prediction_only(
    output_path: str,
    trajectory_id: str,
    baseline_rows: List[Dict[str, Any]],
    oracle_rows: List[Dict[str, Any]],
) -> None:
    """Plot predicted progress curves for baseline and oracle; no ground-truth line."""
    baseline_frame_steps = [
        int(row.get("metadata", {}).get("frame_step", -1)) for row in baseline_rows
    ]
    oracle_frame_steps = [
        int(row.get("metadata", {}).get("frame_step", -1)) for row in oracle_rows
    ]
    if baseline_frame_steps != oracle_frame_steps:
        raise ValueError(f"Frame-step mismatch for trajectory {trajectory_id}")

    video_path = baseline_rows[0].get("video_path")
    frames = load_frames_from_npz(video_path)
    selected_frames = [
        frames[min(max(step, 0), len(frames) - 1)] for step in baseline_frame_steps
    ]

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
    ax_curve.plot(
        x, baseline_curve,
        marker="s", linewidth=2.0,
        color=MODEL_COLORS["baseline"], label="baseline",
    )
    ax_curve.plot(
        x, oracle_curve,
        marker="^", linewidth=2.0,
        color=MODEL_COLORS["oracle"], label="oracle",
    )
    ax_curve.set_xticks(x)
    ax_curve.set_xticklabels(
        [str(step) for step in baseline_frame_steps], rotation=45, ha="right"
    )
    ax_curve.set_xlabel("sampled prefix endpoint (frame_step)")
    ax_curve.set_ylabel("predicted progress")
    ax_curve.set_ylim(-0.03, 1.03)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="best", fontsize=8)
    ax_curve.set_title(trajectory_id, fontsize=10)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _create_trajectory_visualizations(
    comparison_dir: str,
    baseline_grouped: Dict[str, List[Dict[str, Any]]],
    oracle_grouped: Dict[str, List[Dict[str, Any]]],
) -> None:
    trajectory_root = os.path.join(comparison_dir, "trajectory_plots")
    os.makedirs(trajectory_root, exist_ok=True)

    missing_in_oracle = set(baseline_grouped.keys()) - set(oracle_grouped.keys())
    if missing_in_oracle:
        logger.warning(
            f"Trajectories in baseline but not oracle (will skip): {missing_in_oracle}"
        )

    for trajectory_id in sorted(baseline_grouped.keys()):
        if trajectory_id not in oracle_grouped:
            continue
        output_path = os.path.join(trajectory_root, f"{trajectory_id}.png")
        _plot_trajectory_prediction_only(
            output_path=output_path,
            trajectory_id=trajectory_id,
            baseline_rows=baseline_grouped[trajectory_id],
            oracle_rows=oracle_grouped[trajectory_id],
        )


# ---------------------------------------------------------------------------
# Core eval loop
# ---------------------------------------------------------------------------


def _run_dual_model_visualization(
    cfg: BaselineEvalConfig,
    base_data_cfg: DataConfig,
    dataset: Any,
    combined_indices: Dict[str, Any],
    models: Dict[str, Any],
    model_paths: Dict[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Run both models over all trajectories and collect raw predictions."""
    sampler = _build_reward_alignment_sampler(cfg, base_data_cfg, dataset, combined_indices)
    materialized_samples, _ = _build_sampler_materialized_samples(sampler)

    model_results: Dict[str, List[Dict[str, Any]]] = {}
    for scope in MODEL_SCOPES:
        logger.info(f"Running inference for scope: {scope}")
        model_results[scope] = _process_samples(
            cfg=cfg,
            samples=materialized_samples,
            model=models[scope],
            model_scope=scope,
            model_path=model_paths[scope],
        )

    return model_results


# ---------------------------------------------------------------------------
# Output directory helper
# ---------------------------------------------------------------------------


def _build_output_dir(cfg: BaselineEvalConfig) -> str:
    if cfg.output_dir is not None:
        return cfg.output_dir

    normalized_path = _normalize_model_path(cfg.model_path)
    if normalized_path:
        dir_name = f"{cfg.reward_model}_{normalized_path}_pred_visualization_no_labels"
    else:
        dir_name = f"{cfg.reward_model}_pred_visualization_no_labels"
    return os.path.join("./baseline_eval_output", dir_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra_main(version_base=None, config_path="../configs", config_name="baseline_eval_config")
def main(cfg: DictConfig):
    baseline_cfg = convert_hydra_to_dataclass(cfg, BaselineEvalConfig)
    display_config(baseline_cfg)

    if baseline_cfg.reward_model not in [
        "gvl", "vlac", "roboreward", "robodopamine", "topreward", "rbm", "rewind"
    ]:
        raise ValueError(
            "reward_model must be one of: gvl, vlac, roboreward, robodopamine, "
            "topreward, rbm, rewind"
        )

    baseline_cfg.custom_eval.eval_types = ["reward_alignment"]
    baseline_cfg.custom_eval.reward_alignment = [DEFAULT_EVAL_DATASET]
    baseline_cfg.output_dir = _build_output_dir(baseline_cfg)
    os.makedirs(baseline_cfg.output_dir, exist_ok=True)
    logger.info(f"Output directory: {baseline_cfg.output_dir}")

    data_cfg = DataConfig(
        max_frames=baseline_cfg.max_frames,
        load_embeddings=True if "rewind" in baseline_cfg.reward_model else False,
        labeled_progress_data_sources=["gen_progress_success", "gen_progress_failure"],
    )
    display_config(data_cfg)

    dataset, combined_indices, _ = _load_local_processed_dataset(DEFAULT_EVAL_DATASET)
    logger.info(f"Dataset size: {len(dataset)} samples")

    model_paths = {
        "baseline": DEFAULT_BASELINE_MODEL_PATH,
        "oracle": baseline_cfg.model_path,
    }
    models = _initialize_models(baseline_cfg)
    model_results = _run_dual_model_visualization(
        cfg=baseline_cfg,
        base_data_cfg=data_cfg,
        dataset=dataset,
        combined_indices=combined_indices,
        models=models,
        model_paths=model_paths,
    )

    if is_rank_0():
        short_dataset_name = _shorten_dataset_name(DEFAULT_EVAL_DATASET)

        # Save per-scope raw results
        for scope in MODEL_SCOPES:
            scope_dir = os.path.join(baseline_cfg.output_dir, scope)
            os.makedirs(scope_dir, exist_ok=True)
            results_file = os.path.join(scope_dir, f"{short_dataset_name}_results.json")
            with open(results_file, "w") as f:
                json.dump(_make_json_serializable(model_results[scope]), f, indent=2)
            logger.info(f"Saved {scope} results to: {results_file}")

        # Build trajectory-level comparison visualizations
        comparison_dir = os.path.join(baseline_cfg.output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)

        baseline_grouped = _group_results_by_trajectory(model_results["baseline"])
        oracle_grouped = _group_results_by_trajectory(model_results["oracle"])
        _create_trajectory_visualizations(comparison_dir, baseline_grouped, oracle_grouped)
        logger.info(f"Saved trajectory plots to: {comparison_dir}/trajectory_plots/")

        # Write a lightweight run report (config + trajectory list, no metrics)
        run_report = {
            "dataset_name": DEFAULT_EVAL_DATASET,
            "run_timestamp": datetime.now().isoformat(),
            "note": "Visualization-only run — no target_progress labels required, no metrics computed.",
            "config": _make_json_serializable(
                {
                    "reward_model": baseline_cfg.reward_model,
                    "baseline_model_path": model_paths["baseline"],
                    "oracle_model_path": model_paths["oracle"],
                    "max_frames": baseline_cfg.max_frames,
                    "model_config": (
                        asdict(baseline_cfg.model_config)
                        if hasattr(baseline_cfg.model_config, "__dataclass_fields__")
                        else baseline_cfg.model_config
                    ),
                    "custom_eval": (
                        asdict(baseline_cfg.custom_eval)
                        if hasattr(baseline_cfg.custom_eval, "__dataclass_fields__")
                        else baseline_cfg.custom_eval
                    ),
                }
            ),
            "trajectory_ids": sorted(baseline_grouped.keys()),
            "num_trajectories": len(baseline_grouped),
        }
        report_path = os.path.join(comparison_dir, "run_report.json")
        with open(report_path, "w") as f:
            json.dump(run_report, f, indent=2)
        logger.info(f"Saved run report to: {report_path}")

    logger.info("\nPrediction visualization (no-labels) complete!")
    return {
        "baseline_count": len(model_results["baseline"]),
        "oracle_count": len(model_results["oracle"]),
    }


if __name__ == "__main__":
    main()
