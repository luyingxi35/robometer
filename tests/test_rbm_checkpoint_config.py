import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import PretrainedConfig, PreTrainedModel

from robometer.models.rbm import RBM


class RBMCheckpointConfigTest(unittest.TestCase):
    def test_from_pretrained_uses_preloaded_backbone_config(self):
        config = PretrainedConfig()
        base_model = SimpleNamespace(config=config)
        loaded_model = object()

        with patch.object(PreTrainedModel, "from_pretrained", return_value=loaded_model) as parent_loader:
            result = RBM.from_pretrained("checkpoint", base_model=base_model, processor="processor")

        self.assertIs(result, loaded_model)
        self.assertIs(parent_loader.call_args.kwargs["config"], config)
        self.assertIs(parent_loader.call_args.kwargs["base_model"], base_model)

    def test_from_pretrained_rejects_implicit_config_class(self):
        with self.assertRaisesRegex(ValueError, "explicit `config` or a preloaded `base_model`"):
            RBM.from_pretrained("checkpoint")


if __name__ == "__main__":
    unittest.main()
