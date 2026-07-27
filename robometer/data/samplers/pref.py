#!/usr/bin/env python3
"""
PrefSampler class for producing batches of preference data.
"""

from typing import Dict, List, Optional, Any

import random

from robometer.data.dataset_types import PreferenceSample, Trajectory
from robometer.data.samplers.base import RBMBaseSampler
from robometer.data.datasets.helpers import (
    DataGenStrat,
    convert_continuous_to_discrete_bins,
)
from robometer.utils.logger import get_logger, rank_0_info, trace
from robometer.utils.timer import timer

logger = get_logger()


class PrefSampler(RBMBaseSampler):
    """Data generator for producing batches of preference prediction data."""

    def __init__(self, is_evaluation=False, **kwargs):
        super().__init__(**kwargs)

        self.dataset_preference_ratio = self.config.dataset_preference_ratio
        self.preference_strategy_ratio: List[float] = self.config.preference_strategy_ratio
        self._has_suboptimal = (
            any(len(indices) > 0 for indices in self.suboptimal_by_task.values()) if self.suboptimal_by_task else False
        )
        rank_0_info(f"[PREF SAMPLER] Has suboptimal: {self._has_suboptimal}")
        self._build_labeled_progress_indices()

        # Initialize preference dataset
        self._load_preference_dataset()

    def _generate_sample(self, item: dict, preferred_strategy: Optional[DataGenStrat] = None):
        """Generate a preference sample from an item.

        If the item has a non-successful quality label, it will be used as the rejected
        trajectory and an optimal trajectory from the same task will be found as the chosen one.
        Otherwise, normal preference sampling logic is used.

        Args:
            item: The trajectory item
            preferred_strategy: Optional strategy to use (if None, will select strategy based on ratios)
        """
        quality_label = item["quality_label"]
        use_partial_success = item.get("partial_success") is not None

        if self._is_labeled_progress_traj(item) and self._is_labeled_progress_quality(quality_label):
            if quality_label == "failure_labeled":
                raise ValueError(
                    f"Labeled-progress trajectory {item.get('id')} cannot be used as preferred anchor: "
                    "failure_labeled samples are only valid as rejected candidates."
                )
            if quality_label in {"successful_labeled", "suboptimal_labeled"}:
                sample = self._create_labeled_progress_pref_sample(item, preferred_strategy=preferred_strategy)
                if sample is not None:
                    return sample

        # Handle non-successful trajectories: use as rejected, find optimal from same task as chosen
        # skip this for trajectories with partial_success which we will handle with partial success logic
        if quality_label != "successful" and not use_partial_success:
            traj_id = item["id"]
            task_name = item["task"]

            logger.trace(
                f"[PREF SAMPLER] Non-successful quality detected for ID={traj_id}, using as rejected trajectory, task={task_name}"
            )

            # Find optimal trajectories from the same task
            same_task_optimal_indices = self.optimal_by_task.get(task_name, [])

            if not same_task_optimal_indices:
                logger.trace(
                    f"[PREF SAMPLER] No optimal trajectories found for task '{task_name}', falling through to normal sampling"
                )
                return self._create_pref_sample(item, preferred_strategy=preferred_strategy)

            # Select a random optimal trajectory from the same task as chosen
            chosen_idx = random.choice(same_task_optimal_indices)
            chosen_traj_dict = self.dataset[chosen_idx]

            chosen_trajectory = self._get_traj_from_data(chosen_traj_dict)
            rejected_trajectory = self._get_traj_from_data(item)

            sample = PreferenceSample(
                chosen_trajectory=chosen_trajectory,
                rejected_trajectory=rejected_trajectory,
                data_gen_strategy=DataGenStrat.SUBOPTIMAL.value,
            )

            logger.trace(
                f"[PREF SAMPLER] Created preference sample for non-successful traj ID={traj_id} with optimal traj from same task"
            )
            return sample

        return self._create_pref_sample(item, preferred_strategy=preferred_strategy)

    def _create_labeled_progress_pref_sample(
        self, item: Dict[str, Any], preferred_strategy: Optional[DataGenStrat] = None
    ) -> PreferenceSample | None:
        quality_label = item.get("quality_label")
        if quality_label not in self.labeled_quality_order:
            return None

        allowed_quality_labels = {"successful_labeled", "suboptimal_labeled", "failure_labeled"}
        if quality_label not in allowed_quality_labels:
            return None

        if preferred_strategy in {DataGenStrat.REWIND, DataGenStrat.REVERSE_PROGRESS}:
            raise ValueError(
                f"Labeled-progress trajectory {item.get('id')} does not allow rejected strategy "
                f"{preferred_strategy.value}; only same-task quality pairing or different_task are supported."
            )

        strategy_was_sampled = preferred_strategy is None
        if preferred_strategy is None:
            preferred_strategy = self._sample_labeled_progress_pref_strategy()

        if preferred_strategy == DataGenStrat.SUBOPTIMAL:
            sample = self._create_labeled_same_task_pref_sample(item)
            if sample is not None:
                return sample

            sample = self._create_labeled_different_task_pref_sample(item)
            if sample is not None:
                return sample

        if preferred_strategy == DataGenStrat.DIFFERENT_TASK:
            sample = self._create_labeled_different_task_pref_sample(item)
            if sample is not None:
                return sample

            # A sampled different-task strategy can be unavailable for a single-task
            # dataset. Fall back to the other legal strategy instead of randomly
            # failing otherwise valid single-task training items.
            if strategy_was_sampled:
                sample = self._create_labeled_same_task_pref_sample(item)
                if sample is not None:
                    return sample

        if preferred_strategy in {DataGenStrat.SUBOPTIMAL, DataGenStrat.DIFFERENT_TASK}:
            raise ValueError(
                f"Labeled-progress trajectory {item.get('id')} could not form a legal rejected pair: "
                "no allowed same-task failure_labeled match and different_task generation failed."
            )

        raise ValueError(
            f"Labeled-progress trajectory {item.get('id')} received unsupported preferred strategy "
            f"{preferred_strategy.value}; only suboptimal(same-task quality) and different_task are allowed."
        )

    def _sample_labeled_progress_pref_strategy(self) -> DataGenStrat:
        """Sample one of the two legal labeled-progress preference strategies."""
        if len(self.preference_strategy_ratio) < 3:
            raise ValueError(
                "preference_strategy_ratio must contain at least "
                "[rewind, suboptimal_same_task, different_task]."
            )

        same_task_weight = self.preference_strategy_ratio[1]
        different_task_weight = self.preference_strategy_ratio[2]
        if same_task_weight < 0 or different_task_weight < 0:
            raise ValueError(
                "Labeled-progress preference strategy weights must be non-negative: "
                f"suboptimal={same_task_weight}, different_task={different_task_weight}."
            )

        total_weight = same_task_weight + different_task_weight
        if total_weight <= 0:
            raise ValueError(
                "Labeled-progress preference sampling requires a positive weight for "
                "suboptimal_same_task or different_task."
            )

        if self._local_random.random() < same_task_weight / total_weight:
            return DataGenStrat.SUBOPTIMAL
        return DataGenStrat.DIFFERENT_TASK

    def _create_labeled_same_task_pref_sample(
        self, chosen_traj_dict: Dict[str, Any]
    ) -> PreferenceSample | None:
        if chosen_traj_dict.get("quality_label") not in {"successful_labeled", "suboptimal_labeled"}:
            return None

        ref_task_key = self._get_labeled_task_key(chosen_traj_dict)
        failure_indices = self.labeled_failure_indices_by_task_key.get(ref_task_key, [])
        if not failure_indices:
            return None

        rejected_traj_dict = self.dataset[random.choice(failure_indices)]
        chosen_trajectory = self._get_traj_from_data(
            chosen_traj_dict, subsample_strategy="subsample_forward"
        )
        rejected_trajectory = self._get_traj_from_data(
            rejected_traj_dict, subsample_strategy="subsample_forward"
        )
        if chosen_trajectory is None or rejected_trajectory is None:
            return None

        return PreferenceSample(
            chosen_trajectory=chosen_trajectory,
            rejected_trajectory=rejected_trajectory,
            data_gen_strategy=DataGenStrat.SUBOPTIMAL.value,
        )

    def _get_labeled_task_key(self, traj: Dict[str, Any]) -> str:
        """Return a stable task key for labeled-progress grouping.

        Local generated progress datasets vary the natural-language instruction for the
        same environment. Keep that text for prompting, but group samples by the
        trajectory id prefix, e.g. PegInsertionVertical-v1_... -> PegInsertionVertical-v1.
        """
        traj_id = str(traj.get("id", ""))
        if "_" in traj_id:
            return traj_id.split("_", 1)[0]
        return str(traj.get("task", "unknown"))

    def _get_labeled_task_key_from_values(self, traj_id: Any, task: Any) -> str:
        traj_id = str(traj_id or "")
        if "_" in traj_id:
            return traj_id.split("_", 1)[0]
        return str(task or "unknown")

    def _build_labeled_progress_indices(self) -> None:
        self.labeled_indices_by_task_key: Dict[str, List[int]] = {}
        self.labeled_failure_indices_by_task_key: Dict[str, List[int]] = {}

        if not self.labeled_progress_data_sources:
            self.labeled_task_keys = set()
            return

        ids = self.dataset["id"]
        tasks = self.dataset["task"]
        data_sources = self.dataset["data_source"]
        quality_labels = self.dataset["quality_label"]

        for idx, (traj_id, task, data_source, quality_label) in enumerate(
            zip(ids, tasks, data_sources, quality_labels)
        ):
            if data_source not in self.labeled_progress_data_sources:
                continue

            task_key = self._get_labeled_task_key_from_values(traj_id, task)
            self.labeled_indices_by_task_key.setdefault(task_key, []).append(idx)
            if quality_label == "failure_labeled":
                self.labeled_failure_indices_by_task_key.setdefault(task_key, []).append(idx)

        self.labeled_task_keys = set(self.labeled_indices_by_task_key.keys())

    def _get_labeled_same_task_indices(self, ref_traj: Dict[str, Any]) -> List[int]:
        ref_task_key = self._get_labeled_task_key(ref_traj)
        return self.labeled_indices_by_task_key.get(ref_task_key, [])

    def _create_labeled_different_task_pref_sample(self, chosen_traj_dict: Dict[str, Any]) -> PreferenceSample | None:
        chosen_task_key = self._get_labeled_task_key(chosen_traj_dict)
        candidate_task_keys = [task_key for task_key in self.labeled_task_keys if task_key != chosen_task_key]
        if not candidate_task_keys:
            return None

        rejected_task_key = random.choice(candidate_task_keys)
        candidate_indices = self.labeled_indices_by_task_key.get(rejected_task_key, [])
        if not candidate_indices:
            return None

        rejected_traj_dict = self.dataset[random.choice(candidate_indices)]
        chosen_trajectory = self._get_traj_from_data(chosen_traj_dict, subsample_strategy="subsample_forward")
        rejected_trajectory = self._get_traj_from_data(rejected_traj_dict, subsample_strategy="subsample_forward")
        if chosen_trajectory is None or rejected_trajectory is None:
            return None

        rejected_trajectory.target_progress = [0.0] * len(rejected_trajectory.target_progress)
        if self.config.progress_loss_type.lower() == "discrete":
            rejected_trajectory.target_progress = convert_continuous_to_discrete_bins(
                rejected_trajectory.target_progress, self.config.progress_discrete_bins
            )
        if rejected_trajectory.success_label is not None:
            rejected_trajectory.success_label = [0.0] * len(rejected_trajectory.success_label)

        return PreferenceSample(
            chosen_trajectory=chosen_trajectory,
            rejected_trajectory=rejected_trajectory,
            data_gen_strategy=DataGenStrat.DIFFERENT_TASK.value,
        )

    def _execute_strategy(
        self, strategy: DataGenStrat, chosen_traj: Dict[str, Any], use_partial_success: bool
    ) -> tuple[Dict[str, Any], str, Dict[str, Any]] | None:
        """Execute a strategy to get rejected trajectory.

        Args:
            strategy: The strategy to execute
            chosen_traj: The chosen trajectory
            use_partial_success: Whether this trajectory uses partial_success

        Returns:
            Tuple of (rejected_traj, rejected_subsample_strategy, chosen_traj) or None if failed
            Note: chosen_traj may be swapped with rejected_traj for partial_success trajectories
        """
        max_retries = 3
        rejected_subsample_strategy = None
        rejected_traj = None

        if strategy == DataGenStrat.REWIND:
            rejected_traj = chosen_traj.copy()
            rejected_subsample_strategy = "subsample_rewind"
        elif strategy == DataGenStrat.SUBOPTIMAL:
            for _ in range(max_retries):
                rejected_traj = self._get_same_task_suboptimal(chosen_traj)
                if rejected_traj is not None:
                    # For trajectories with partial_success, if the returned trajectory has higher partial_success, swap them
                    if use_partial_success:
                        chosen_partial_success = chosen_traj.get("partial_success")
                        rejected_partial_success = rejected_traj.get("partial_success")
                        if rejected_partial_success is not None and chosen_partial_success is not None:
                            if rejected_partial_success > chosen_partial_success:
                                logger.trace(
                                    f"[PREF SAMPLER] Swapping trajectories: found higher partial_success "
                                    f"({rejected_partial_success} > {chosen_partial_success})"
                                )
                                rejected_traj, chosen_traj = chosen_traj, rejected_traj
                    break
            rejected_subsample_strategy = "subsample_forward"
        elif strategy == DataGenStrat.DIFFERENT_TASK:
            for _ in range(max_retries):
                rejected_traj = self._get_different_video_traj(chosen_traj)
                if rejected_traj is not None:
                    break
            rejected_subsample_strategy = "subsample_forward"
        elif strategy == DataGenStrat.REVERSE_PROGRESS:
            rejected_traj = chosen_traj.copy()
            rejected_subsample_strategy = "subsample_reverse"
        else:
            return None

        if rejected_traj is None:
            return None

        return (rejected_traj, rejected_subsample_strategy, chosen_traj)

    def _create_pref_sample_from_dataset(self) -> PreferenceSample:
        """Create a preference sample from the loaded preference dataset."""
        if not self.preferences:
            return None

        # For now, return a simple preference sample
        # This can be enhanced later when we have actual preference data
        random.choice(self.preferences)

        # This is a placeholder - would need to be implemented based on actual preference data structure
        return None

    def _load_preference_dataset(self):
        """Load the preference dataset from disk or hub if provided."""
        self.preferences = []

        # For now, we'll use empty preferences since the config structure has changed
        # This can be updated later if needed
        rank_0_info("[PREF SAMPLER] No preference dataset provided, will use random sampling for preferences")
        return

    def _create_preference_sample(self) -> PreferenceSample:
        """Create a preference prediction sample: chosen vs rejected where chosen is preferred.
        Either from dataset or from generated trajectories.

        Returns:
            PreferenceSample: A preference sample with chosen (preferred) vs rejected
            (suboptimal) trajectories and associated metadata
        """

        with timer("create_preference_sample", verbose=False):
            if random.random() < self.dataset_preference_ratio and self.preferences:
                # Use preference trajectories from dataset
                return self._create_pref_sample_from_dataset()
            else:
                return self._create_pref_sample()

    def _create_pref_sample(
        self, chosen_traj: Optional[Dict[str, Any]] = None, preferred_strategy: Optional[DataGenStrat] = None
    ) -> PreferenceSample:
        """Create a preference prediction sample using various rejected trajectory generation strategies.

        Rewind Same Task
        - Creates a suboptimal trajectory by rewinding the chosen trajectory

        Suboptimal/Failure Same Task
        - Uses existing suboptimal/failure trajectories from the same task

        Different Task
        - Uses trajectories from completely different tasks

        Returns:
            PreferenceSample: A preference sample with chosen (preferred) vs rejected
            (suboptimal) trajectories and associated metadata

        Raises:
            ValueError: If no chosen trajectories are available for preference generation
            RuntimeError: If all strategies fail and fallback rewind also fails
        """
        # Log when preference sampler is called
        traj_id = chosen_traj["id"] if chosen_traj is not None else "sampling_new"
        logger.trace(f"[PREF SAMPLER] Creating preference sample for trajectory ID: {traj_id}")

        # Use provided chosen trajectory if given; otherwise sample one
        if chosen_traj is None:
            # Use preprocessed chosen trajectories from index maps
            if not self.optimal_by_task:
                return None

            # Filter out tasks with empty optimal_indices to avoid infinite loop
            valid_tasks = {
                task: indices
                for task, indices in self.optimal_by_task.items()
                if indices  # Only include tasks with non-empty indices
            }

            if not valid_tasks:
                # No valid tasks with optimal trajectories available
                return None

            # Get a random task and chosen trajectory from it
            task_name = random.choice(list(valid_tasks.keys()))
            optimal_indices = valid_tasks[task_name]

            # Double-check that we have valid indices (should always be true now)
            if not optimal_indices:
                return None

            chosen_idx = random.choice(optimal_indices)
            chosen_traj = self.dataset[chosen_idx]

        # Initialize variables for strategy selection
        rejected_traj = None
        strategy_used = None
        rejected_subsample_strategy = None

        # Check if this trajectory uses partial_success
        use_partial_success = chosen_traj.get("partial_success") is not None
        if use_partial_success:
            partial_success = chosen_traj.get("partial_success")
            logger.trace(
                f"[PREF SAMPLER] Trajectory with partial_success detected (ID: {chosen_traj.get('id', 'unknown')}, partial_success: {partial_success})"
            )

        # Strategy selection: use preferred_strategy if provided, otherwise select based on ratios
        if preferred_strategy is not None:
            # Use the preferred strategy directly
            logger.trace(f"[PREF SAMPLER] Using preferred strategy: {preferred_strategy.value}")
            result = self._execute_strategy(preferred_strategy, chosen_traj, use_partial_success)
            if result is None:
                logger.trace(f"[PREF SAMPLER] Preferred strategy {preferred_strategy.value} failed, returning None")
                return None
            rejected_traj, rejected_subsample_strategy, chosen_traj = result
            strategy_used = preferred_strategy
            attempt = 1  # Set attempt for preferred strategy path
        else:
            # Strategy selection with rebalancing on failure
            strategies = []
            if self.preference_strategy_ratio[0] > 0:
                if not (
                    getattr(self.config, "labeled_progress_disable_rewind", True) and self._is_labeled_progress_traj(chosen_traj)
                ):
                    strategies.append((DataGenStrat.REWIND, self.preference_strategy_ratio[0]))
            if self._has_suboptimal and self.preference_strategy_ratio[1] > 0:
                strategies.append((DataGenStrat.SUBOPTIMAL, self.preference_strategy_ratio[1]))
            if self.preference_strategy_ratio[2] > 0:
                strategies.append((DataGenStrat.DIFFERENT_TASK, self.preference_strategy_ratio[2]))
            if self.preference_strategy_ratio[3] > 0:
                strategies.append((DataGenStrat.REVERSE_PROGRESS, self.preference_strategy_ratio[3]))

            max_attempts = 10  # Limit retry attempts to prevent infinite loops
            max_strategy_attempts = 3  # Maximum attempts per strategy before removing it
            attempt = 0

            # Track attempts per strategy
            strategy_attempt_counts = {strat: 0 for strat, _ in strategies}

            while rejected_traj is None and attempt < max_attempts:
                attempt += 1

                # Check if we have any strategies left
                if not strategies:
                    return None

                # Rebalance probabilities based on remaining strategies
                total_prob = sum(prob for _, prob in strategies)
                if total_prob == 0:
                    return None

                # Normalize probabilities
                normalized_strategies = [(strat, prob / total_prob) for strat, prob in strategies]

                # Select strategy based on rebalanced probabilities
                prob = random.random()
                cumulative_prob = 0.0
                selected_strategy = None

                for strat, normalized_prob in normalized_strategies:
                    cumulative_prob += normalized_prob
                    if prob <= cumulative_prob:
                        selected_strategy = strat
                        break

                # Log strategy attempt
                logger.trace(
                    f"[PREF SAMPLER] Attempt {attempt}/{max_attempts}: Trying strategy {selected_strategy.value if selected_strategy else 'None'}"
                )

                # Execute selected strategy
                result = self._execute_strategy(selected_strategy, chosen_traj, use_partial_success)
                if result is not None:
                    rejected_traj, rejected_subsample_strategy, chosen_traj = result
                    strategy_used = selected_strategy
                    logger.trace(f"[PREF SAMPLER] Strategy {selected_strategy.value} succeeded on attempt {attempt}")
                else:
                    # Strategy failed - increment attempt count
                    strategy_attempt_counts[selected_strategy] = strategy_attempt_counts.get(selected_strategy, 0) + 1
                    failed_count = strategy_attempt_counts[selected_strategy]

                    logger.trace(
                        f"[PREF SAMPLER] Strategy {selected_strategy.value} failed (failure count: {failed_count}/{max_strategy_attempts})"
                    )

                    # Only remove strategy if it has failed max_strategy_attempts times
                    if strategy_attempt_counts[selected_strategy] >= max_strategy_attempts:
                        logger.trace(
                            f"[PREF SAMPLER] Removing strategy {selected_strategy.value} after {max_strategy_attempts} consecutive failures"
                        )
                        strategies = [(strat, prob) for strat, prob in strategies if strat != selected_strategy]
                        continue

            # If we still don't have a sample after all attempts, return None
            if rejected_traj is None:
                logger.trace(
                    f"[PREF SAMPLER] Failed to generate preference sample after {max_attempts} attempts - all strategies exhausted"
                )
                return None

        chosen_subsample_strategy = "subsample_forward"
        chosen_trajectory = self._get_traj_from_data(chosen_traj, subsample_strategy=chosen_subsample_strategy)

        rejected_trajectory = self._get_traj_from_data(rejected_traj, subsample_strategy=rejected_subsample_strategy)

        if rejected_trajectory is None or chosen_trajectory is None:
            return None

        # If our strategy is different task, make sure the rejected trajectory has 0 progress and 0 success labels
        if strategy_used in [
            DataGenStrat.DIFFERENT_TASK,
            DataGenStrat.DIFFERENT_TASK_INSTRUCTION,
        ]:
            rejected_trajectory.target_progress = [0.0] * len(rejected_trajectory.target_progress)
            if self.config.progress_loss_type.lower() == "discrete":
                rejected_trajectory.target_progress = convert_continuous_to_discrete_bins(
                    rejected_trajectory.target_progress, self.config.progress_discrete_bins
                )
            # Also set success labels to 0.0 (predict 0 success for different task trajectories)
            if rejected_trajectory.success_label is not None:
                rejected_trajectory.success_label = [0.0] * len(rejected_trajectory.success_label)

        # Create preference sample structure
        sample = PreferenceSample(
            chosen_trajectory=chosen_trajectory,
            rejected_trajectory=rejected_trajectory,
            data_gen_strategy=strategy_used.value,
        )
        sample.resample_attempts = attempt
        return sample
