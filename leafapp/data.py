from datasets import load_dataset

DATASET_NAME = "DScomp380/plant_village"

def load_subset(fraction, seed):
    """Return a random fraction of the plant_village rows, shuffled with seed."""
    ds = load_dataset(DATASET_NAME, split="train")
    n_rows = int(fraction * len(ds))
    return ds.shuffle(seed=seed).select(range(n_rows))