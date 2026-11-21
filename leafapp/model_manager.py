import gc
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping
from leafapp.predict import LeafClassifier

class ModelManager:
    """Load models on demand and retain only the active classifier."""

    def __init__(
        self,
        model_dirs: Mapping[str, str | Path],
        loader: Callable[[Path], Any] | None = None,
    ):
        self.model_dirs = {
            name: Path(path)
            for name, path in model_dirs.items()
        }

        self._loader = loader if loader is not None else LeafClassifier.from_dir
        self._active_model_name = None
        self._active_classifier = None
        self._lock = Lock()

    def top_k(self, model_name, image, k=5):
        """Predict using the selected model, loading it when necessary."""

        if model_name not in self.model_dirs:
            raise ValueError(f"Unknown model: {model_name}")

        with self._lock:
            if self._active_model_name != model_name:
                self._active_classifier = None
                self._active_model_name = None
                gc.collect()
                classifier = self._loader(self.model_dirs[model_name])

                self._active_classifier = classifier
                self._active_model_name = model_name

            return self._active_classifier.top_k(image, k=k)