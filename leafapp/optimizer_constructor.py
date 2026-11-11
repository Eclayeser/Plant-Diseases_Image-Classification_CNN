import torch

OPTIMIZERS = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
}

def choose_optimizer(parameters, train_settings):
    """Build the optimiser named in the train settings (AdamW if not given)"""

    name = train_settings.get("optimizer", "AdamW")
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimiser '{name}'. Choose from: {list(OPTIMIZERS)}")
    optimizer_class = OPTIMIZERS[name]
    return optimizer_class(
        parameters,
        lr=train_settings["lr"],
        weight_decay=train_settings["weight_decay"]
    )