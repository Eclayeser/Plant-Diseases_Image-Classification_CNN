from typing import Any

from leafapp.models.cnn import BasicCNN

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
