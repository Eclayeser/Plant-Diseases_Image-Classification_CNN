import json

from leafapp.config import MODEL_FILE, LABELS_FILE, MODEL_CONFIG_FILE
from scripts import fetch_model_from_clearml as fetch

def create_fake_package(directory, model_id):
    """Create a fake ClearML model package without downloading anything."""
    directory.mkdir(parents=True)

    (directory / MODEL_FILE).write_bytes(model_id.encode())
    (directory / LABELS_FILE).write_text('["healthy"]')
    (directory / MODEL_CONFIG_FILE).write_text(
        json.dumps({"image_size": 224})
    )

    return directory

def configure_fetch(tmp_path, monkeypatch, registry):
    """Redirect the downloader to temporary test directories."""
    lock_file = tmp_path / "model.lock.json"
    lock_file.write_text(json.dumps(registry))

    destination = tmp_path / "deployment" / "model"

    monkeypatch.setattr(fetch, "LOCK_FILE", lock_file)
    monkeypatch.setattr(fetch, "MODEL_DIR", destination)

    return destination

def test_legacy_single_model_download(tmp_path, monkeypatch):
    """Single-model deployments retain the original directory structure."""
    destination = configure_fetch(
        tmp_path,
        monkeypatch,
        {"model_id": "model-a"},
    )

    package = create_fake_package(tmp_path / "package-a", "model-a")
    downloaded = []

    def fake_download(model_id):
        downloaded.append(model_id)
        return package

    monkeypatch.setattr(
        fetch,
        "download_onnx_model_from_clearml",
        fake_download,
    )

    fetch.main()

    assert downloaded == ["model-a"]
    assert (destination / MODEL_FILE).read_bytes() == b"model-a"
    assert (destination / LABELS_FILE).exists()
    assert (destination / MODEL_CONFIG_FILE).exists()

def test_download_three_models_separately(tmp_path, monkeypatch):
    """Each registered model is downloaded into its own directory."""
    registry = {
        "model_id": "model-a",
        "models": [
            {"name": "Model 1", "model_id": "model-a"},
            {"name": "Model 2", "model_id": "model-b"},
            {"name": "Model 3", "model_id": "model-c"},
        ],
    }

    destination = configure_fetch(
        tmp_path,
        monkeypatch,
        registry,
    )

    packages = {
        model_id: create_fake_package(tmp_path / f"package-{model_id}", model_id,)
        for model_id in ("model-a", "model-b", "model-c")
    }

    downloaded = []
    def fake_download(model_id):
        downloaded.append(model_id)
        return packages[model_id]

    monkeypatch.setattr(
        fetch,
        "download_onnx_model_from_clearml",
        fake_download,
    )
    fetch.main()

    assert downloaded == ["model-a", "model-b", "model-c"]
    for index, model_id in enumerate(downloaded):
        model_dir = destination / f"model_{index}"
        assert (model_dir / MODEL_FILE).read_bytes() == model_id.encode()
        assert (model_dir / LABELS_FILE).exists()
        assert (model_dir / MODEL_CONFIG_FILE).exists()

    assert not (destination / MODEL_FILE).exists()