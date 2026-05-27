import unittest
from unittest.mock import patch

from robometer.configs.experiment_configs import DataConfig
from robometer.data.datasets.helpers import DataGenStrat, compute_success_labels
from robometer.data.datasets.rbm_data import RBMDataset
from robometer.data.samplers.pref import PrefSampler
from robometer.data.dataset_types import Trajectory


class FakeHFDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


class CountingFakeHFDataset(FakeHFDataset):
    def __init__(self, rows):
        super().__init__(rows)
        self.row_access_count = 0

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        self.row_access_count += 1
        return super().__getitem__(key)


class TestLabeledPrefSampler(unittest.TestCase):
    def _make_config(self):
        cfg = DataConfig()
        cfg.labeled_progress_data_sources = ["labeled_source"]
        cfg.preference_strategy_ratio = [1, 1, 1, 1]
        cfg.progress_loss_type = "l2"
        cfg.sample_type_ratio = [1, 0, 0]
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
        if isinstance(dataset, list):
            dataset = FakeHFDataset(dataset)
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

    def test_training_anchor_indices_exclude_failure_labeled(self):
        dataset_rows = [
            {"id": "succ", "task": "peg", "data_source": "labeled_source", "quality_label": "successful_labeled", "is_robot": True},
            {"id": "fail", "task": "peg", "data_source": "labeled_source", "quality_label": "failure_labeled", "is_robot": True},
            {"id": "other", "task": "peg", "data_source": "other_source", "quality_label": "failure", "is_robot": True},
        ]
        rbm_dataset = object.__new__(RBMDataset)
        rbm_dataset.dataset = dataset_rows
        rbm_dataset.config = self._make_config()
        rbm_dataset.is_evaluation = False

        anchor_indices = RBMDataset._build_anchor_indices(rbm_dataset)

        self.assertEqual(anchor_indices, [0, 2])

    def test_eval_anchor_indices_keep_failure_labeled(self):
        dataset_rows = [
            {"id": "succ", "task": "peg", "data_source": "labeled_source", "quality_label": "successful_labeled", "is_robot": True},
            {"id": "fail", "task": "peg", "data_source": "labeled_source", "quality_label": "failure_labeled", "is_robot": True},
        ]
        rbm_dataset = object.__new__(RBMDataset)
        rbm_dataset.dataset = dataset_rows
        rbm_dataset.config = self._make_config()
        rbm_dataset.is_evaluation = True

        anchor_indices = RBMDataset._build_anchor_indices(rbm_dataset)

        self.assertEqual(anchor_indices, [0, 1])

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

    def test_labeled_same_task_uses_trajectory_id_prefix_not_instruction_text(self):
        dataset = [
            {
                "id": "PegInsertionVertical-v1_offset_reach_position_2442",
                "task": "Raise the insertion peg, and guide it upright into the hole.",
                "data_source": "labeled_source",
                "quality_label": "suboptimal_labeled",
                "is_robot": True,
            },
            {
                "id": "PegInsertionVertical-v1_fail_to_grasp_grasp_3",
                "task": "Grab the workpiece peg, and place it straight downward through the hole opening.",
                "data_source": "labeled_source",
                "quality_label": "failure_labeled",
                "is_robot": True,
            },
        ]
        sampler = self._make_sampler(dataset, self._make_combined_indices({"unused": [0, 1]}))
        sampler._get_traj_from_data = self._fake_get_traj

        sample = sampler._create_labeled_progress_pref_sample(dataset[0])

        self.assertEqual(sample.chosen_trajectory.id, "PegInsertionVertical-v1_offset_reach_position_2442")
        self.assertEqual(sample.rejected_trajectory.id, "PegInsertionVertical-v1_fail_to_grasp_grasp_3")

    def test_labeled_pref_does_not_scan_entire_dataset_per_sample(self):
        dataset = CountingFakeHFDataset(
            [
                {
                    "id": "PegInsertionVertical-v1_offset_reach_position_2442",
                    "task": "Raise the insertion peg, and guide it upright into the hole.",
                    "data_source": "labeled_source",
                    "quality_label": "suboptimal_labeled",
                    "is_robot": True,
                },
                {
                    "id": "PegInsertionVertical-v1_fail_to_grasp_grasp_3",
                    "task": "Grab the workpiece peg, and place it straight downward through the hole opening.",
                    "data_source": "labeled_source",
                    "quality_label": "failure_labeled",
                    "is_robot": True,
                },
                {
                    "id": "OtherTask-v1_success_0",
                    "task": "Other task",
                    "data_source": "labeled_source",
                    "quality_label": "successful_labeled",
                    "is_robot": True,
                },
            ]
        )
        sampler = self._make_sampler(dataset, self._make_combined_indices({"unused": [0, 1, 2]}))
        sampler._get_traj_from_data = self._fake_get_traj

        dataset.row_access_count = 0
        sample = sampler._create_labeled_progress_pref_sample(dataset[0])

        self.assertEqual(sample.rejected_trajectory.id, "PegInsertionVertical-v1_fail_to_grasp_grasp_3")
        self.assertLessEqual(
            dataset.row_access_count,
            2,
            "Labeled preference sampling should use prebuilt indices instead of scanning all rows per sample.",
        )

    def test_failure_labeled_cannot_be_used_as_preferred_anchor(self):
        dataset = [
            {"id": "fail", "task": "peg", "data_source": "labeled_source", "quality_label": "failure_labeled", "is_robot": True},
        ]
        sampler = self._make_sampler(dataset, self._make_combined_indices({"peg": [0]}))

        with self.assertRaisesRegex(ValueError, "cannot be used as preferred anchor"):
            sampler._generate_sample(dataset[0])

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
