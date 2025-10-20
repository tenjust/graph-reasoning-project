import json
from pathlib import Path
from typing import Iterable

import h5py


def iterate_rebel_hdf5_split(path: Path, split: str, batch_size: int = 1000) -> Iterable[dict]:
    """
    Load REBEL dataset from HDF5 file in batches, yielding one example at a time.
    Each element is a JSON string decoded into a Python dict.
    :param path: Path to HDF5 file.
    :param split: Split name ('train', 'validation', 'test').
    :param batch_size: Number of examples to read in each batch.
    :return: Iterator of dicts each representing one data instance.
    """
    if split not in ('train', 'validation', 'test'):
        raise ValueError(f"Invalid split name: {split}. Must be 'train', 'validation', or 'test'.")
    with h5py.File(path, "r") as f:
        dset = f[split]
        total = dset.shape[0]
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = dset[start:end]
            for x in batch:
                yield json.loads(x.decode("utf-8"))


def load_hdf5_data(path, split="train", num_samples=None) -> list[dict]:
    with h5py.File(path, "r") as f:
        data = f[split]
        samples = []
        for i, item in enumerate(data):
            if num_samples and i >= num_samples:
                break
            sample_json = json.loads(item.decode("utf-8"))
            samples.append(sample_json)
    return samples