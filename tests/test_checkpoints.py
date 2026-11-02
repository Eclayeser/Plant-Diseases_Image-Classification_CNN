# Checks that checkpoints save and load a model without losing anything.
import torch
from leafapp.models.cnn import BasicCNN
from leafapp.checkpoints import save_checkpoint, load_checkpoint


def test_checkpoint_round_trip_gives_same_outputs(tmp_path):
    cfg = {"model": {"channels": [8]}, "data": {"image_size": 32}}
    class_names = ["a", "b", "c"]
    model = BasicCNN(num_classes=len(class_names), channels=cfg["model"]["channels"], image_size=cfg["data"]["image_size"])
    model.eval()

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, cfg, class_names)
    loaded_model, _, _ = load_checkpoint(path)

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(loaded_model(x), model(x)), "Loaded model gives different outputs"

def test_checkpoint_keeps_config_and_class_names(tmp_path):
    cfg = {"model": {"channels": [8]}, "data": {"image_size": 32}, "seed": 42}
    class_names = ["a", "b", "c"]
    model = BasicCNN(num_classes=len(class_names), channels=cfg["model"]["channels"], image_size=cfg["data"]["image_size"])

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, cfg, class_names)
    _, loaded_names, loaded_config = load_checkpoint(path)

    assert loaded_names == class_names, "Class names changed after loading"
    assert loaded_config == cfg, "Config changed after loading"