"""Save example leaf images for the Gradio gallery."""

from leafapp.config import ROOT
from leafapp.data import load_subset, split

classes = [
    "Tomato___healthy",
    "Tomato___Leaf_Mold",
    "Apple___Apple_scab",
    "Corn___Common_rust",
    "Potato___Late_blight",
    "Grape___Black_rot",
]

folder = ROOT / "app" / "examples"
folder.mkdir(parents=True, exist_ok=True)

# Use the same subset and split as training, so the model never saw these images
data = load_subset(fraction=0.2, seed=42)
train, val, test = split(data, seed=42)

class_names = test.features["label"].names

# Save one image from each class
for name in classes:
    label = class_names.index(name)

    for row in test:
        if row["label"] == label:
            filename = name.replace("___", "_").replace("(", "").replace(")", "").lower() + ".jpg"
            row["image"].convert("RGB").save(folder / filename)
            print("Saved:", filename)
            break