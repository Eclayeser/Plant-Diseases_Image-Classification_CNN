# Holds settings shared across data preparation, training, evaluation and prediction.
import yaml
from typing import Any

from leafapp.models.cnn import BasicCNN

def load_config(path):
    """
    Load a YAML config file and return it as a dict.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)      # turns the YAML into a normal Python dict
    return cfg


def build_model(model_settings: dict[str, Any], data_specs: dict[str, Any], class_names: list[str]):
    model = None

    if model_settings.get("type", None) is None:
        raise ValueError("Failed to construct a model. Config file should contain model type.")

    if model_settings["type"] == "BasicCNN":
        model = BasicCNN(
            num_classes=len(class_names),
            channels=model_settings["channels"],
            image_size=data_specs["image_size"],
        )

    # when new model is implemented, wire its construction here via if statement!

    if model is None:
        raise ValueError(f"Failed to construct a model. Config file specifies {model_settings['type']} - unidentified type.")

    return model
