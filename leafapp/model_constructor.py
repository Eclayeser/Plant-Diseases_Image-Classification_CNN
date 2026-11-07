from typing import Any

from leafapp.models.cnn import BasicCNN
from leafapp.models.resnet import build_resnet18
from leafapp.models.regularised_cnn import RegularisedCNN

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
    if model_settings["type"] == "RegularisedCNN":
        model = RegularisedCNN(
            num_classes=len(class_names),
            channels=model_settings["channels"],
            image_size=data_specs["image_size"],
            regularisation=model_settings["regularisation"],
            dropout_p=model_settings.get("dropout_p", 0.2),
        )

    if model_settings["type"] == "ResNet18":
        model = build_resnet18(num_classes=len(class_names))

    if model is None:
        raise ValueError(f"Failed to construct a model. Config file specifies {model_settings['type']} - unidentified type.")

    return model
