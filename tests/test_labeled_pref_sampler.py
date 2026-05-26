import unittest
from unittest.mock import patch

from robometer.configs.experiment_configs import DataConfig
from robometer.data.datasets.helpers import DataGenStrat, compute_success_labels
from robometer.data.samplers.pref import PrefSampler
from robometer.data.dataset_types import Trajectory


class TestLabeledPrefSampler(unittest.TestCase):
    def _make_config(self):
        cfg = DataConfig()
        cfg.labeled_progress_data_sources = ["labeled_source"]
        cfg.preference_strategy_ratio = [1, 1, 1, 1]
        cfg.progress_loss_type = "l2"
        return cfg

    def _make_combined_indices(self, task_indices, optimal_by_task=None, suboptimal_by_task=None):
        return {
            "robot_trajectories": [],
            "human_trajectories": [],
            "optimal_by_task": optimal_by_task or {},
            "suboptimal_by_task": suboptimal_by_task or {},
            "quality_indices": {},
            "task_indices": task_indices,
            "source_indices": {},
            "partial_success_indices": {},
            "paired_human_robot_by_task": {},
            "tasks_with_multiple_quality_labels": [],
        }

    def _make_sampler(self, dataset, combined_indices):
        sampler = PrefSampler(
            config=self._make_config(),
            dataset=dataset,
            combined_indices=combined_indices,
            dataset_success_cutoff_map={},
            verbose=False,
            random_seed=0,
            is_evaluation=False,
        )
        return sampler

    def _fake_get_traj(self, traj_dict, subsample_strategy=None):
        return Trajectory(
            id=traj_dict.get("id"),
            task=traj_dict.get("task"),
            data_source=traj_dict.get("data_source"),
            quality_label=traj_dict.get("quality_label"),
            target_progress=[0.0, 1.0],
            success_label=[0.0, 1.0],
            frames_shape=(2, 224, 224, 3),
        )

    def test_labeled_success_labels_use_progress_threshold(self):
        labels = compute_success_labels(
            target_progress=[0.0, 0.4, 1.0],
            data_source="labeled_source",
            dataset_success_percent={},
            max_success=1.0,
            quality_label="successful_labeled",
            labeled_progress_data_sources=["labeled_source"],
            labeled_quality_order={"successful_labeled": 2, "suboptimal_labeled": 1, "failure_labeled": 0},
        )
        self.assertEqual(labels, [0.0, 0.0, 1.0])

    def test_labeled_success_labels_all_zero_below_threshold(self):
        labels = compute_success_labels(
            target_progress=[0.0, 0.5, 0.9],
            data_source="labeled_source",
            dataset_success_percent={},
            max_success=1.0,
            quality_label="failure_labeled",
            labeled_progress_data_sources=["labeled_source"],
            labeled_quality_order={"successful_labeled": 2, "suboptimal_labeled": 1, "failure_labeled": 0},
        )
        self.assertEqual(labels, [0.0, 0.0, 0.0])

    def test_success_labeled_pairs_only_with_failure_labeled(self):
        dataset = [
            {"id": "succ", "task": "peg", "data_source": "labeled_source", "quality_label": "successful_labeled", "is_robot": True},
            {"id": "sub", "task": "peg", "data_source": "labeled_source", "quality_label": "suboptimal_labeled", "is_robot": True},
            {"id": "fail", "task": "peg", "data_source": "labeled_source", "quality_label": "failure_labeled", "is_robot": True},
        ]
        sampler = self._make_sampler(dataset, self._make_combined_indices({"peg": [0, 1, 2]}))
        sampler._get_traj_from_data = self._fake_get_traj

        with patch("random.choice", side_effect=lambda seq: seq[0]):
            sample = sampler._create_labeled_progress_pref_sample(dataset[0])

        self.assertEqual(sample.chosen_trajectory.id, "succ")
        self.assertEqual(sample.rejected_trajectory.id, "fail")
        self.assertEqual(sample.data_gen_strategy, DataGenStrat.SUBOPTIMAL.value)

    def test_suboptimal_labeled_pairs_only_with_failure_labeled(self):
        dataset = [
            {"id": "sub", "task": "peg", "data_source": "labeled_source", "quality_label": "suboptimal_labeled", "is_robot": True},
            {"id": "fail", "task": "peg", "data_source": "labeled_source", "quality_label": "failure_labeled", "is_robot": True},
        ]
        sampler = self._make_sampler(dataset, self._make_combined_indices({"peg": [0, 1]}))
        sampler._get_traj_from_data = self._fake_get_traj

        sample = sampler._create_labeled_progress_pref_sample(dataset[0])

        self.assertEqual(sample.chosen_trajectory.id, "sub")
        self.assertEqual(sample.rejected_trajectory.id, "fail")

    def test_failure_labeled_falls_back_to_different_task(self):
        dataset = [
            {"id": "fail", "task": "peg", "data_source": "labeled_source", "quality_label": "failure_labeled", "is_robot": True},
        ]
        sampler = self._make_sampler(dataset, self._make_combined_indices({"peg": [0]}))
        fallback = object()
        sampler._create_pref_sample = lambda item, preferred_strategy=None: fallback

        sample = sampler._create_labeled_progress_pref_sample(dataset[0])

        self.assertIs(sample, fallback)

    def test_raises_when_no_failure_match_and_different_task_fails(self):
        dataset = [
            {"id": "succ", "task": "peg", "data_source": "labeled_source", "quality_label": "successful_labeled", "is_robot": True},
            {"id": "sub", "task": "peg", "data_source": "labeled_source", "quality_label": "suboptimal_labeled", "is_robot": True},
        ]
        sampler = self._make_sampler(dataset, self._make_combined_indices({"peg": [0, 1]}))
        sampler._create_pref_sample = lambda item, preferred_strategy=None: None

        with self.assertRaisesRegex(ValueError, "could not form a legal rejected pair"):
            sampler._create_labeled_progress_pref_sample(dataset[0])

    def test_rewind_strategy_rejected_for_labeled_data(self):
        dataset = [
            {"id": "succ", "task": "peg", "data_source": "labeled_source", "quality_label": "successful_labeled", "is_robot": True},
            {"id": "fail", "task": "peg", "data_source": "labeled_source", "quality_label": "failure_labeled", "is_robot": True},
        ]
        sampler = self._make_sampler(dataset, self._make_combined_indices({"peg": [0, 1]}))

        with self.assertRaisesRegex(ValueError, "does not allow rejected strategy"):
            sampler._create_labeled_progress_pref_sample(dataset[0], preferred_strategy=DataGenStrat.REWIND)


if __name__ == "__main__":
    unittest.main()
