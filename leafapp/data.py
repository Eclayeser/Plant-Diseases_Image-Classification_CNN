from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

DATASET_NAME = "DScomp380/plant_village"


def load_subset(fraction, seed):
    """Return a random fraction of the plant_village rows, shuffled with seed."""
    ds = load_dataset(DATASET_NAME, split="train")
    n_rows = int(fraction * len(ds))
    return ds.shuffle(seed=seed).select(range(n_rows))


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


def make_transforms(image_size, augment=True):
    """Build the image transforms for training and evaluation.

    Return (train_tf, eval_tf). Both resize images to image_size and
    convert them to tensors. train_tf also adds random flips, rotations
    and colour changes when augment is True.
    """
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    if augment:
        train_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, saturation=0.2),
                transforms.ToTensor(),
            ]
        )
    else:
        train_tf = eval_tf
    return train_tf, eval_tf
