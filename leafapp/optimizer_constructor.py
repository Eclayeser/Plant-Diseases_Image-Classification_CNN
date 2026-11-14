import torch

OPTIMIZERS = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
}

def choose_optimizer(parameters, train_settings):
    """Build the optimiser named in the train settings (AdamW if not given).

    The chosen name is written back into train_settings, so it is recorded in
    ClearML and in the checkpoint even when the config doesn't set it.
    """

    name = train_settings.setdefault("optimizer", "AdamW")
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimiser '{name}'. Choose from: {list(OPTIMIZERS)}")
    optimizer_class = OPTIMIZERS[name]
    return optimizer_class(
        parameters,
        lr=train_settings["lr"],
        weight_decay=train_settings["weight_decay"]
    )