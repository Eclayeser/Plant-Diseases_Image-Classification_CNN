from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "deployment" / "model"
MODEL_FILE = "model.onnx"
LABELS_FILE = "labels.json"
MODEL_CONFIG_FILE = "model_config.json"


def load_config(path):
    """
    Load a YAML config file and return it as a dict.
    """
    import yaml  # important to keep inside the function; the deployed app without pyyaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)      # turns the YAML into a normal Python dict
    return cfg
