import argparse
import json
from pathlib import Path

import torch
from leafapp.checkpoints import load_checkpoint


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "deployment"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export a trained CNN to ONNX.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained .pt checkpoint.",
    )
    return parser.parse_args(argv)


def main(checkpoint_path):
    model, class_names, _, config = load_checkpoint(checkpoint_path)
    model.eval()

    image_size = config["data"]["image_size"]
    dummy_input = torch.randn(1, 3, image_size, image_size)

    OUTPUT_DIR.mkdir(exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_DIR / "model.onnx"),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )

    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(class_names, f, indent=2)

    with open(OUTPUT_DIR / "model_config.json", "w") as f:
        json.dump({
            "image_size": image_size,
            "normalize": config["data"].get("normalize", False),
        }, f, indent=2)

    print(f"ONNX export completed from {checkpoint_path}")
    print(f"Classes: {len(class_names)}")
    print(f"Image size: {image_size}")


if __name__ == "__main__":
    args = parse_args()
    main(args.checkpoint)