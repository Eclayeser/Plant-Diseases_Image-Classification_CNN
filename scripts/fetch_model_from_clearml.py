import json
import shutil

from leafapp.config import (
    LABELS_FILE,
    MODEL_CONFIG_FILE,
    MODEL_DIR,
    MODEL_FILE,
    ROOT,
)
from leafapp.tracking import download_onnx_model_from_clearml

LOCK_FILE = ROOT / "deployment" / "model.lock.json"


def main():
    """Download the ClearML model pinned in model.lock.json into MODEL_DIR."""
    model_id = json.loads(LOCK_FILE.read_text(encoding="utf-8"))["model_id"]
    package_dir = download_onnx_model_from_clearml(model_id)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for file_name in (MODEL_FILE, LABELS_FILE, MODEL_CONFIG_FILE):
        shutil.copy2(package_dir / file_name, MODEL_DIR / file_name)
    print(f"Downloaded ClearML model {model_id} to {MODEL_DIR}")


if __name__ == "__main__":
    main()
