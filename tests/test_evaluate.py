# Checks that evaluation metrics are calculated correctly.
import pytest
from evaluate import parse_args

def test_parse_args_default_split():
    args = parse_args(["--checkpoint", "model.pt"])
    assert args.checkpoint == "model.pt", "Did not read --checkpoint"
    assert args.split == "validation", "Default split should be validation"

def test_parse_args_custom_split():
    args = parse_args(["--checkpoint", "model.pt", "--split", "test"])
    assert args.split == "test", "Did not read --split"

def test_parse_args_requires_checkpoint():
    with pytest.raises(SystemExit):
        parse_args([])

def test_parse_args_rejects_invalid_split():
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "model.pt", "--split", "train"])