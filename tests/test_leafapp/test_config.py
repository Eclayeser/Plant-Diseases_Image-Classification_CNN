import yaml

from leafapp.config import load_config
from tests.shared_helpers import make_train_cfg


def test_load_config_reads_yaml_back_into_the_same_dict(tmp_path):
    cfg = make_train_cfg(tmp_path / "checkpoints")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    assert load_config(path) == cfg, "Loaded config should match the YAML that was written"
