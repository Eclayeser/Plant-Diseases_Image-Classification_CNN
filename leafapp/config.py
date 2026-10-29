# Holds settings shared across data preparation, training, evaluation and prediction.
import yaml

def load_config(path):
    """
    Load a YAML config file and return it as a dict.
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)      # turns the YAML into a normal Python dict
    return cfg