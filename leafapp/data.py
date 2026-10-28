from datasets import load_dataset

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
