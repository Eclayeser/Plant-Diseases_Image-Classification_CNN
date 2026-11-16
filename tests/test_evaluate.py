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

def test_parse_args_requires_checkpoint():
    with pytest.raises(SystemExit):
        parse_args(["--responsible", VALID_NAME])

def test_parse_args_rejects_invalid_split():
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "model.pt", "--split", "train", "--responsible", VALID_NAME])