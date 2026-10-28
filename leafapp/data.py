from datasets import load_dataset

DATASET_NAME = "DScomp380/plant_village"


def load_subset(fraction, seed):
    """Return a random fraction of the plant_village rows, shuffled with seed."""
    ds = load_dataset(DATASET_NAME, split="train")
    n_rows = int(fraction * len(ds))
    return ds.shuffle(seed=seed).select(range(n_rows))


def split(ds, seed, val_frac=0.15, test_frac=0.15):
    cut1 = ds.train_test_split(
        test_size=val_frac + test_frac,
        seed=seed,
        stratify_by_column="label",
    )
    cut2 = cut1["test"].train_test_split(
        test_size=0.5,
        seed=seed,
        stratify_by_column="label",
    )
    return cut1["train"], cut2["train"], cut2["test"]
