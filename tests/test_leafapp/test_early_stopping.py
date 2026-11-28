from leafapp.early_stopping import EarlyStopping


def test_stops_after_patience_epochs_without_improvement():
    """Training stops once the loss fails to improve for patience epochs in a row."""
    stopper = EarlyStopping(patience=2)
    # 0.85 is a new best after a bad epoch, so the counter has to reset there
    answers = [stopper.step(loss) for loss in [1.0, 0.9, 0.95, 0.85, 0.9, 0.95]]
    expected = [False, False, False, False, False, True]
    assert (
        answers == expected
    ), "Should stop exactly when 2 epochs in a row fail to improve"


def test_small_improvement_does_not_reset_counter():
    """A drop smaller than min_delta does not count as an improvement."""
    stopper = EarlyStopping(patience=2, min_delta=0.1)
    # each loss is lower than the last, but never more than 0.1 below the best (1.0)
    answers = [stopper.step(loss) for loss in [1.0, 0.95, 0.92]]
    expected = [False, False, True]
    assert (
        answers == expected
    ), "Should stop exactly when 2 epochs in a row improve by less than min_delta"
