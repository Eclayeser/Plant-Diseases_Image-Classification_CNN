import json

from huggingface_hub import snapshot_download

from leafapp.config import (
    LABELS_FILE,
    MODEL_CONFIG_FILE,
    MODEL_DIR,
    MODEL_FILE,
    ROOT,
)

LOCK_FILE = ROOT / "deployment" / "model.lock.json"


def main():
    """Download the model version pinned in model.lock.json into MODEL_DIR."""
    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    snapshot_download(
        repo_id=lock["repo_id"],
        revision=lock["revision"],
        local_dir=MODEL_DIR,
        allow_patterns=[MODEL_FILE, LABELS_FILE, MODEL_CONFIG_FILE],
    )
    print(f"Downloaded {lock['repo_id']} at {lock['revision'][:7]} to {MODEL_DIR}")


if __name__ == "__main__":
    main()
