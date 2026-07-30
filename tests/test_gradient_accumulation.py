import ast
import unittest
from pathlib import Path

class TestGradientAccumulation(unittest.TestCase):
    def _training_step_ast(self):
        trainer_path = (
            Path(__file__).resolve().parents[1]
            / "robometer"
            / "trainers"
            / "rbm_heads_trainer.py"
        )
        tree = ast.parse(trainer_path.read_text())
        trainer_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RBMHeadsTrainer"
        )
        training_step = next(
            node
            for node in trainer_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "training_step"
        )
        return training_step

    def test_training_step_does_not_clear_accumulated_gradients(self):
        training_step = self._training_step_ast()
        zero_grad_calls = [
            node
            for node in ast.walk(training_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "zero_grad"
        ]

        self.assertEqual(
            zero_grad_calls,
            [],
            "RBMHeadsTrainer.training_step must leave gradient clearing to "
            "Transformers Trainer so gradient_accumulation_steps > 1 works.",
        )

    def test_training_step_logs_only_when_gradients_are_synchronized(self):
        training_step = self._training_step_ast()
        sync_gradient_reads = [
            node
            for node in ast.walk(training_step)
            if isinstance(node, ast.Attribute)
            and node.attr == "sync_gradients"
        ]

        self.assertTrue(
            sync_gradient_reads,
            "Per-step metric reduction must be gated by accelerator.sync_gradients "
            "so it runs once per optimizer step, not once per micro-step.",
        )


if __name__ == "__main__":
    unittest.main()
