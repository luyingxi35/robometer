import unittest
from types import SimpleNamespace

import torch

from robometer.trainers.rbm_heads_trainer import RBMHeadsTrainer


class SuccessLossEmptyMaskTest(unittest.TestCase):
    def test_empty_supervision_mask_returns_differentiable_zero(self):
        trainer = object.__new__(RBMHeadsTrainer)
        trainer.config = SimpleNamespace(
            data=SimpleNamespace(min_success=0.5, use_multi_image=True),
            loss=SimpleNamespace(
                progress_loss_type="discrete",
                progress_discrete_bins=10,
                success_balance_classes=True,
                success_positive_weight=1.0,
                success_pairwise_weight=0.0,
            ),
            model=SimpleNamespace(base_model_id="Qwen"),
        )
        logits = torch.tensor([[0.2, -0.3]], requires_grad=True)
        progress = torch.tensor([[5.0, 7.0]])
        labels = torch.zeros_like(logits)

        loss, accuracy, auprc, _ = trainer._compute_success_loss_helper(
            logits, progress, labels
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(accuracy.item(), 0.0)
        self.assertEqual(auprc.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


if __name__ == "__main__":
    unittest.main()
