import shutil

from leafapp.config import (
    LABELS_FILE,
    MODEL_CONFIG_FILE,
    MODEL_DIR,
    MODEL_FILE,
    ROOT,
)
from leafapp.model_registry import load_model_registry
from leafapp.tracking import download_onnx_model_from_clearml

LOCK_FILE = ROOT / "deployment" / "model.lock.json"


def download_model(model_id, destination):
    package_dir = download_onnx_model_from_clearml(model_id)
    destination.mkdir(parents=True, exist_ok=True)

    for file_name in (MODEL_FILE, LABELS_FILE, MODEL_CONFIG_FILE):
        shutil.copy2(
            package_dir / file_name,
            destination / file_name,
        )

    print(f"Downloaded ClearML model {model_id} to {destination}")


def main():
    registry = load_model_registry(LOCK_FILE)
    models = registry["models"]

    if len(models) == 1:
        download_model(
            models[0]["model_id"],
            MODEL_DIR,
        )
        return

    for index, model in enumerate(models):
        destination = MODEL_DIR / f"model_{index}"

        download_model(
            model["model_id"],
            destination,
        )

if __name__ == "__main__":
    main()