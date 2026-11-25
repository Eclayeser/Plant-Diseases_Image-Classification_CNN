# Runs the whole pipeline once on tiny fake data: train, evaluate, export to ONNX and predict.
from unittest.mock import MagicMock

from PIL import Image

import evaluate
import train
from leafapp.predict import LeafClassifier
from scripts import export_onnx
from utils import make_fake_loader


CLASS_NAMES = ["a", "b", "c", "d", "e"]


def fake_get_loaders(**kwargs):
    return make_fake_loader(), make_fake_loader(), make_fake_loader(), CLASS_NAMES


def test_end_to_end_pipeline(monkeypatch, tmp_path):
    cfg = {
        "seed": 0,
        "data": {
            "fraction": 1.0,
            "image_size": 32,
            "batch_size": 4,
            "normalize": "none",
            "augment": False,
        },
        "model": {
            "type": "BasicCNN",
            "channels": [8],
        },
        "train": {
            "lr": 0.001,
            "weight_decay": 0.0,
            "epochs": 1,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }

    # Use fake data and configuration
    monkeypatch.setattr(train, "load_config", lambda path: cfg)
    monkeypatch.setattr(train, "get_loaders", fake_get_loaders)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)

    # Fake ClearML
    task = MagicMock()
    monkeypatch.setattr(train, "initialise_task", lambda **kwargs: (task, cfg))
    monkeypatch.setattr(train, "upload_checkpoint", MagicMock())
    monkeypatch.setattr(train, "close_task", MagicMock())

    # Train the model
    train.main("fake.yaml", train.NAMES[0], "debugging", "")

    # Find the saved checkpoint
    checkpoints = list((tmp_path / "checkpoints").glob("*_best.pt"))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]

    # 2. Evaluate the trained model
    monkeypatch.setattr(evaluate, "get_loaders", fake_get_loaders)
    monkeypatch.setattr(evaluate, "initialise_task", lambda **kwargs: (task, kwargs["config"]))
    monkeypatch.setattr(evaluate, "log_final_metrics", MagicMock())
    monkeypatch.setattr(evaluate, "log_image", MagicMock())
    monkeypatch.setattr(evaluate, "upload_file", MagicMock())
    monkeypatch.setattr(evaluate, "close_task", MagicMock())

    # Save evaluation outputs in a temporary folder
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(evaluate, "DIAGRAMS_DIR", tmp_path / "diagrams")

    evaluate.main(checkpoint, "test", train.NAMES[0], "debugging", "", None, None)

    # Check the evaluation report
    reports = list((tmp_path / "reports").glob("*_report.txt"))
    assert len(reports) == 1
    assert "Top-1 Accuracy" in reports[0].read_text()


    # Export the trained checkpoint to ONNX without ClearML
    monkeypatch.setattr(export_onnx, "get_task", lambda task_id: task)
    monkeypatch.setattr(export_onnx, "download_checkpoint", lambda task, artifact: checkpoint)

    model_dir = tmp_path / "model"
    export_onnx.main("fake-task-id", out_dir=model_dir)

    # Load the exported model and predict one image
    classifier = LeafClassifier.from_dir(model_dir)
    predictions = classifier.top_k(Image.new("RGB", (64, 64), "green"), k=5)

    # Check the top-5 predictions
    assert len(predictions) == 5
    for name, probability in predictions:
        assert name in CLASS_NAMES
        assert 0 <= probability <= 1