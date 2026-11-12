from datasets import ClassLabel, Dataset
from PIL import Image
import torch
from leafapp.data import make_batch_transform, make_transforms, split


def test_transforms_give_right_shape_and_range():
    """Both transforms give a [3, size, size] tensor with values in [0, 1]."""
    photo = Image.new("RGB", (256, 256), color=(120, 200, 80))
    train_tf, eval_tf = make_transforms(64)
    for tf in (train_tf, eval_tf):
        x = tf(photo)
        assert x.shape == (3, 64, 64)
        assert x.min().item() >= 0.0 and x.max().item() <= 1.0


def test_no_augment_reuses_eval_transform():
    """With augment=False, the training transform is the evaluation transform."""
    train_tf, eval_tf = make_transforms(64, augment=False)
    assert train_tf is eval_tf


def test_batch_transform_converts_each_photo():
    """The batch transform converts every photo and keeps the labels."""
    photo = Image.new("RGB", (256, 256), color=(120, 200, 80))
    _, eval_tf = make_transforms(64)
    transform_batch = make_batch_transform(eval_tf)
    out = transform_batch({"image": [photo, photo], "label": [3, 7]})
    assert len(out["pixels"]) == 2
    assert all(x.shape == (3, 64, 64) for x in out["pixels"])
    assert out["label"] == [3, 7]


def test_split_sizes_no_overlap_and_all_labels():
    """The split gives 70/15/15, shares no rows, and keeps every label in every set."""
    ds = Dataset.from_dict(
        {"id": list(range(100)), "label": [i % 4 for i in range(100)]}
    )
    ds = ds.cast_column("label", ClassLabel(names=["a", "b", "c", "d"]))
    train, val, test = split(ds, seed=0)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    train_ids, val_ids, test_ids = (set(p["id"]) for p in (train, val, test))
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
    for pile in (train, val, test):
        assert set(pile["label"]) == {0, 1, 2, 3}


def test_normalize_is_applied_once_without_augment():
    """With augment off, train_tf and eval_tf normalise the photo exactly once."""
    photo = Image.new("RGB", (8, 8), color=(102, 204, 51))
    train_tf, eval_tf = make_transforms(8, augment=False, normalize=True)
    assert torch.equal(train_tf(photo), eval_tf(photo)), "train_tf and eval_tf differ"
    assert torch.allclose(
        eval_tf(photo)[:, 0, 0], torch.tensor([-0.37, 1.54, -0.92]), atol=0.01
    ), "The photo was not normalised exactly once"


def test_normalize_off_keeps_totensor_values():
    """The default, normalize=False, leaves ToTensor's 0 to 1 values unchanged."""
    photo = Image.new("RGB", (8, 8), color=(102, 204, 51))
    _, eval_tf = make_transforms(8)
    assert torch.allclose(
        eval_tf(photo)[:, 0, 0], torch.tensor([0.4, 0.8, 0.2]), atol=0.01
    ), "normalize=False should not change the values"