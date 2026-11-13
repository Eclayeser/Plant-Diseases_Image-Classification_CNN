
import json
from pathlib import Path

import torch
from leafapp.checkpoints import load_checkpoint

# Run this script from the repository root.
CHECKPOINT = (
    "outputs/models/"
    "basiccnn_20261109-115505_74f4b7.pt"
)

OUTPUT_DIR = Path("deployment")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load your team's actual trained CNN.
model, class_names, _, config = load_checkpoint(CHECKPOINT)
model.eval()

image_size = config["data"]["image_size"]

# Dummy input defines the expected image dimensions.
dummy_input = torch.randn(1, 3, image_size, image_size)

# Export architecture and trained weights together.
torch.onnx.export(
    model,
    dummy_input,
    str(OUTPUT_DIR / "model.onnx"),
    input_names=["image"],
    output_names=["logits"],
    opset_version=17,
    dynamo=False,
)

# Preserve original class ordering.
with open(OUTPUT_DIR / "labels.json", "w") as f:
    json.dump(class_names, f, indent=2)

# Preserve preprocessing configuration.
with open(OUTPUT_DIR / "model_config.json", "w") as f:
    json.dump({
        "image_size": image_size,
        "normalize": config["data"].get("normalize", False)
    }, f, indent=2)

print("ONNX export completed.")
print(f"Classes: {len(class_names)}")
print(f"Image size: {image_size}")