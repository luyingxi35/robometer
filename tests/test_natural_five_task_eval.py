import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "eval_natural_five_task",
    Path(__file__).resolve().parents[1] / "evals" / "eval_natural_five_task.py",
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_step_and_discover_checkpoints(tmp_path):
    for name in ["checkpoint-100", "checkpoint-50", "not-a-checkpoint"]:
        (tmp_path / name).mkdir()

    assert mod.step_from_checkpoint(tmp_path / "checkpoint-100") == 100
    assert [p.name for p in mod.discover_checkpoints(tmp_path)] == ["checkpoint-50", "checkpoint-100"]


def test_safe_corr_handles_ties_and_constants():
    assert mod.safe_corr([1, 2, 3], [1, 4, 9], ranked=True) == 1.0
    assert mod.safe_corr([1, 1, 1], [1, 2, 3], ranked=True) is None


def test_failure_metrics_counts():
    records = [
        {"quality_label": "failure_labeled", "success_prob": 0.1},
        {"quality_label": "failure_labeled", "success_prob": 0.9},
        {"quality_label": "successful_labeled", "success_prob": 0.8},
        {"quality_label": "successful_labeled", "success_prob": 0.2},
    ]

    out = mod.failure_metrics(records, threshold=0.5)

    assert out["tp"] == 1
    assert out["tn"] == 1
    assert out["fp"] == 1
    assert out["fn"] == 1
    assert out["balanced_accuracy"] == 0.5


def test_build_metrics_includes_three_families():
    records = []
    for task in ["A", "B"]:
        records.extend([
            {
                "scope": task,
                "quality_label": "successful_labeled",
                "target_progress": [0.0, 1.0],
                "pred_progress": [0.0, 1.0],
                "success_prob": 0.9,
            },
            {
                "scope": task,
                "quality_label": "failure_labeled",
                "target_progress": [0.0, 0.2],
                "pred_progress": [0.0, 0.1],
                "success_prob": 0.1,
            },
        ])

    metrics = mod.build_metrics(records, step=10, model_path="/m/checkpoint-10", top_k=1, threshold=0.5)

    families = {m["family"] for m in metrics}
    assert families == {"progress", "failure_detection", "filtering"}
    assert any(m["family"] == "filtering" and m["scope"] == "task_macro" for m in metrics)
