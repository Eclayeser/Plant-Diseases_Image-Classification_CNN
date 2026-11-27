from pathlib import Path

import pytest
import evaluate
from evaluate import parse_args
from tests.shared_helpers import VALID_NAME


def parse(*args):
    """parse_args with a valid --responsible already filled in."""
    return parse_args(["--responsible", VALID_NAME, *args])


def test_parse_args_default_split():
    args = parse("--checkpoint", "model.pt")
    assert args.checkpoint == "model.pt", "Did not read --checkpoint"
    assert args.split == "validation", "Default split should be validation"


def test_parse_args_custom_split():
    args = parse("--checkpoint", "model.pt", "--split", "test")
    assert args.split == "test", "Did not read --split"


def test_parse_args_rejects_invalid_split():
    with pytest.raises(SystemExit):
        parse("--checkpoint", "model.pt", "--split", "train")


def test_parse_args_requires_checkpoint_or_task_id():
    with pytest.raises(SystemExit):
        parse()


def test_parse_args_rejects_checkpoint_and_task_id_together():
    with pytest.raises(SystemExit):
        parse("--checkpoint", "model.pt", "--task-id", "abc123")


def test_parse_args_task_id_defaults_to_best_checkpoint():
    args = parse("--task-id", "abc123")
    assert args.artifact == "best_checkpoint", "Should default to best_checkpoint with --task-id"


def test_parse_args_task_id_keeps_chosen_artifact():
    args = parse("--task-id", "abc123", "--artifact", "final_checkpoint")
    assert args.artifact == "final_checkpoint", "Should not overwrite the user's choice"


def test_parse_args_rejects_artifact_without_task_id():
    with pytest.raises(SystemExit):
        parse("--checkpoint", "model.pt", "--artifact", "final_checkpoint")


def test_parse_args_rejects_invalid_artifact():
    with pytest.raises(SystemExit):
        parse("--task-id", "abc123", "--artifact", "best")


@pytest.mark.parametrize("split", ["validation", "test"])
def test_report_formats_metrics_and_relative_diagram_paths(monkeypatch, tmp_path, split):
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    diagrams = Path("outputs") / "diagrams"
    metrics = {
        "accuracy": 0.81, "top-3-accuracy": 0.93, "top-5-accuracy": 0.98,
        "precision": 0.72, "recall": 0.63, "f1": 0.67,
        "per-class-metrics": ({}, "healthy  0.84\nrust  0.71"),
    }
    report = evaluate.format_evaluation_report(
        "best.pt", split, metrics, [(3, "rust", "healthy")],
        curve_path=tmp_path / diagrams / "curve.png",
        matrix_path=tmp_path / diagrams / "matrix.png",
        grid_path=tmp_path / diagrams / "grid.png",
    )

    assert report.splitlines() == [
        f"---   Checkpoint: best.pt | Data Set Chosen: {split.upper()}   ---",
        "- Top-1 Accuracy: 0.81", "- Top-3 Accuracy: 0.93", "- Top-5 Accuracy: 0.98",
        "- Macro Precision: 0.72", "- Macro Recall: 0.63", "- Macro F1: 0.67",
        "", " Per-Class Metrics:", "healthy  0.84", "rust  0.71",
        " Most Confused Pairs (true -> predicted):", " - rust -> healthy: 3",
        "", "==============================   Diagrams   ==============================",
        f" - Training Curve (Historical Metrics): {diagrams / 'curve.png'}",
        f" - Confusion Matrix: {diagrams / 'matrix.png'}",
        f" - Misclassified Images: {diagrams / 'grid.png'}",
    ]
