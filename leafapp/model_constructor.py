from typing import Any

from leafapp.models.cnn import BasicCNN
from leafapp.models.resnet import build_resnet18
from leafapp.models.regularised_cnn import RegularisedCNN

def build_model(model_settings: dict[str, Any], data_specs: dict[str, Any], class_names: list[str], pretrained: bool = True):
    model_type = model_settings.get("type")
    if model_type is None:
        raise ValueError("Failed to construct a model. Config file should contain model type.")

    if model_type == "BasicCNN":
        return BasicCNN(
            num_classes=len(class_names),
            channels=model_settings["channels"],
            image_size=data_specs["image_size"],
            pooling=model_settings.get("pooling", "flatten"),
        )

    if model_type == "RegularisedCNN":
        return RegularisedCNN(
            num_classes=len(class_names),
            channels=model_settings["channels"],
            image_size=data_specs["image_size"],
            regularisation=model_settings["regularisation"],
            dropout_p=model_settings.get("dropout_p", 0.2),
            pooling=model_settings.get("pooling", "flatten"),
        )

    if model_type == "ResNet18":
        return build_resnet18(num_classes=len(class_names), pretrained=pretrained)

    raise ValueError(f"Failed to construct a model. Config file specifies {model_type} - unidentified type.")
