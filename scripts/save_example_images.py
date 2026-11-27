"""
Save example leaf images for the Gradio gallery
and the per-class examples shown after a prediction.
"""

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


def main():
    folder = ROOT / "app" / "gallery"
    folder.mkdir(parents=True, exist_ok=True)

    # Take the images from the test split of the same 20% subset used in our experiments
    data = load_subset(fraction=0.2, seed=42)
    train, val, test = split(data, seed=42)

    class_names = test.features["label"].names

    # Save one image from each class
    for name in classes:
        label = class_names.index(name)

        for row in test:
            if row["label"] == label:
                filename = name.replace("___", "_").lower() + ".jpg"
                row["image"].convert("RGB").save(folder / filename)
                print("Saved:", filename)
                break


def save_class_examples(per_class=3):
    data = load_subset(fraction=0.2, seed=42)
    train, _, _ = split(data, seed=42)  # Train split

    class_names = train.features["label"].names
    labels = train["label"]

    for label, name in enumerate(class_names):
        if name == "Background_without_leaves":
            continue

        indices = []
        for i in range(len(labels)):
            if labels[i] == label:
                indices.append(i)
            if len(indices) == per_class:
                break

        class_folder = ROOT / "app" / "class_examples" / name
        class_folder.mkdir(parents=True, exist_ok=True)

        num = 0
        for index in indices:
            num += 1
            train[index]["image"].convert("RGB").save(class_folder / f"{num}.jpg")
            
        print("Saved:", name)


if __name__ == "__main__":
    main()
    save_class_examples()