from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

DATASET_NAME = "DScomp380/plant_village"
DEFAULT_AUGMENTATION = {
    "flip_probability": 0.5,
    "rotation_degrees": 15,
    "brightness": 0.2,
    "saturation": 0.2,
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PLANT_VILLAGE_MEAN = [0.4671, 0.4897, 0.4125]
PLANT_VILLAGE_STD = [0.1914, 0.1669, 0.2109]
NORMALIZATION_STATS = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "plant_village": (PLANT_VILLAGE_MEAN, PLANT_VILLAGE_STD),
}


def load_subset(fraction, seed):
    """Return a stratified random fraction of the plant_village rows.

    Every label keeps its share of the full dataset. A fraction of 1
    returns the whole dataset, shuffled with seed.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"Invalid fraction {fraction}: expected a value in (0, 1]")
    ds = load_dataset(DATASET_NAME, split="train")
    if fraction < 1:
        subset = ds.train_test_split(
            train_size=fraction,
            seed=seed,
            stratify_by_column="label",
        )
        return subset["train"]
    return ds.shuffle(seed=seed)


def split(ds, seed, val_frac=0.15, test_frac=0.15):
    """Split ds by label into train, validation and test sets.

    Return (train, val, test). Keep the fractions the same across all
    runs you compare.
    """
    held_out = val_frac + test_frac
    cut1 = ds.train_test_split(
        test_size=held_out,
        seed=seed,
        stratify_by_column="label",
    )
    cut2 = cut1["test"].train_test_split(
        test_size=test_frac / held_out,
        seed=seed,
        stratify_by_column="label",
    )
    train, val, test = cut1["train"], cut2["train"], cut2["test"]
    return train, val, test


def get_normalization_stats(normalize):
    """Return the (mean, std) pair that a normalize setting selects.

    normalize is "imagenet" (or True) or "plant_village". False returns
    None (no normalising); any other value raises a ValueError.
    """
    if normalize is not False:
        stats = "imagenet" if normalize is True else normalize
        if stats not in NORMALIZATION_STATS:
            raise ValueError(
                f"Invalid normalize {normalize!r}: expected True, False, "
                "'imagenet' or 'plant_village'"
            )
        return NORMALIZATION_STATS[stats]
    return None


def resolve_augmentation(overrides=None):
    """Return the training augmentation settings with any experiment overrides."""
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("data.augmentation must be a mapping of transform settings")
    unknown = overrides.keys() - DEFAULT_AUGMENTATION.keys()
    if unknown:
        raise ValueError(f"Unknown augmentation settings: {unknown}")
    settings = DEFAULT_AUGMENTATION | overrides
    if not 0 <= settings["flip_probability"] <= 1:
        raise ValueError("flip_probability must be between 0 and 1")
    return settings


def make_transforms(image_size, augment=True, normalize=False, *, augmentation=None):
    """Build the image transforms for training and evaluation.

    Return (train_tf, eval_tf). Both resize images to image_size and
    convert them to tensors. train_tf also adds random flips, rotations
    and colour changes when augment is True. Both normalise with the
    stats that get_normalization_stats(normalize) selects.
    """
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    if augment:
        settings = resolve_augmentation(augmentation)
        train_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=settings["flip_probability"]),
                transforms.RandomRotation(settings["rotation_degrees"]),
                transforms.ColorJitter(brightness=settings["brightness"], saturation=settings["saturation"]),
                transforms.ToTensor(),
            ]
        )
    else:
        train_tf = eval_tf

    stats = get_normalization_stats(normalize)
    if stats is not None:
        mean, std = stats
        normalization = transforms.Normalize(mean, std)
        eval_tf = transforms.Compose([eval_tf, normalization])
        train_tf = transforms.Compose([train_tf, normalization])
    return train_tf, eval_tf


def make_batch_transform(image_transform):
    """Return a function that converts a batch of rows with image_transform."""

    def transform_batch(batch):
        """Convert the photos in batch to tensors and keep their labels."""
        return {
            "pixels": [image_transform(img.convert("RGB")) for img in batch["image"]],
            "label": batch["label"],
        }

    return transform_batch


def get_loaders(fraction=0.2, image_size=128, batch_size=32, seed=42, augment=True,
                normalize=False, *, augmentation=None):
    """Build the train, validation and test loaders for plant_village.

    Return (train_loader, val_loader, test_loader, class_names). Each
    batch is a dict: "pixels" of shape [batch_size, 3, image_size,
    image_size] and "label" of shape [batch_size]. class_names[i] is the
    name of label i.
    """
    rows = load_subset(fraction, seed)
    train, val, test = split(rows, seed)
    train_tf, eval_tf = make_transforms(image_size, augment, normalize, augmentation=augmentation)

    train.set_transform(make_batch_transform(train_tf))
    val.set_transform(make_batch_transform(eval_tf))
    test.set_transform(make_batch_transform(eval_tf))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    class_names = rows.features["label"].names

    return train_loader, val_loader, test_loader, class_names
