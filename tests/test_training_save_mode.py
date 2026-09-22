import unittest

from robometer.configs.experiment_configs import TrainingConfig
from robometer.utils.setup_utils import create_training_arguments


class TrainingSaveModeTest(unittest.TestCase):
    def test_save_only_model_reaches_transformers_arguments(self):
        config = TrainingConfig(save_only_model=True)
        arguments = create_training_arguments(config, "/tmp/robometer-test-output")

        self.assertTrue(arguments.save_only_model)


if __name__ == "__main__":
    unittest.main()
