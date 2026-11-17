# Checks that evaluate.py reads its command-line arguments correctly.
import pytest
from evaluate import parse_args
from leafapp.tracking import NAMES

VALID_NAME = NAMES[0]

def test_parse_args_default_split():
    args = parse_args(["--checkpoint", "model.pt", "--responsible", VALID_NAME])
    assert args.checkpoint == "model.pt", "Did not read --checkpoint"
    assert args.split == "validation", "Default split should be validation"


def test_parse_args_custom_split():
    args = parse_args(["--checkpoint", "model.pt", "--split", "test", "--responsible", VALID_NAME])
    assert args.split == "test", "Did not read --split"


def test_parse_args_rejects_invalid_split():
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "model.pt", "--split", "train", "--responsible", VALID_NAME])


def test_parse_args_requires_checkpoint_or_task_id():
    with pytest.raises(SystemExit):
        parse_args(["--responsible", VALID_NAME])


def test_parse_args_rejects_checkpoint_and_task_id_together():
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "model.pt", "--task-id", "abc123", "--responsible", VALID_NAME])


def test_parse_args_task_id_defaults_to_best_checkpoint():
    args = parse_args(["--task-id", "abc123", "--responsible", VALID_NAME])
    assert args.artifact == "best_checkpoint", "Should default to best_checkpoint with --task-id"


def test_parse_args_task_id_keeps_chosen_artifact():
    args = parse_args(["--task-id", "abc123", "--artifact", "final_checkpoint", "--responsible", VALID_NAME])
    assert args.artifact == "final_checkpoint", "Should not overwrite the user's choice"


def test_parse_args_rejects_artifact_without_task_id():
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "model.pt", "--artifact", "final_checkpoint", "--responsible", VALID_NAME])


def test_parse_args_rejects_invalid_artifact():
    with pytest.raises(SystemExit):
        parse_args(["--task-id", "abc123", "--artifact", "best", "--responsible", VALID_NAME])