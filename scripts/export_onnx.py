import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
import numpy as np
import onnx
import onnxruntime

from leafapp.checkpoints import load_checkpoint
from leafapp.config import (
    MODEL_DIR,
    MODEL_CONFIG_FILE,
    MODEL_FILE,
    LABELS_FILE,
)
from leafapp.data import get_normalization_stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export a trained CNN to ONNX.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained .pt checkpoint.",
    )
    parser.add_argument(
        "--out",
        default=MODEL_DIR,
        help="Folder to write the ONNX model, labels and config to.",
    )
    return parser.parse_args(argv)


def build_export_model(model, normalize):
    """Return the model to export: normalisation is prepended, so the graph takes 0-1 RGB pixels."""
    stats = get_normalization_stats(normalize)
    if stats is not None:
        mean, std = stats
        model = nn.Sequential(transforms.Normalize(mean, std), model)

    model.eval()
    return model


def check_parity(torch_model, onnx_path, sample, atol=1e-4):
    """Raise ValueError if ONNX Runtime's output for sample differs from PyTorch's."""
    with torch.no_grad():
        expected = torch_model(sample).numpy()
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": sample.numpy()})[0]
    if not np.allclose(expected, actual, atol=atol):
        raise ValueError(f"ONNX output differs from PyTorch by {np.abs(expected - actual).max():.2e}")


def main(checkpoint_path, out_dir=MODEL_DIR):
    model, class_names, _, config = load_checkpoint(checkpoint_path)
    image_size = config["data"]["image_size"]
    normalize = config["data"].get("normalize", False)

    export_model = build_export_model(model, normalize)
    sample = torch.rand(1, 3, image_size, image_size)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    onnx_path = out_dir / MODEL_FILE

    torch.onnx.export(
        export_model,
        sample,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))
    check_parity(export_model, onnx_path, sample)

    with open(out_dir / LABELS_FILE, "w") as f:
        json.dump(class_names, f, indent=2)

    with open(out_dir / MODEL_CONFIG_FILE, "w") as f:
        json.dump({
            "image_size": image_size,
            "normalization_in_graph": normalize,
            "source_checkpoint": Path(checkpoint_path).name,
        }, f, indent=2)

    print(f"ONNX export completed from {checkpoint_path} to {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.checkpoint, args.out)
    