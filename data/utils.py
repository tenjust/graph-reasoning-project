from __future__ import annotations
from typing import Iterable
from pathlib import Path
import json
import numpy as np
import h5py

from data.amr_datasets import RebelAMRDataset


def decode_bytes_json(x) -> dict:
    """Decode one HDF5 element stored as JSON bytes into a Python dict."""
    if isinstance(x, (bytes, bytearray)):
        b = bytes(x)
    else:
        try:
            b = x.tobytes()  # numpy/h5py scalar case
        except AttributeError:
            b = bytes(x)
    return json.loads(b.decode("utf-8"))

def encode_json_obj(obj: dict) -> bytes:
    """Serialize a Python dict to UTF-8 JSON bytes."""
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def load_original_split(h5_path: Path, split: str, n: int) -> list[dict]:
    """Load up to n JSON-decoded records from a given split."""
    out = []
    with h5py.File(h5_path, "r")as f:
        if split not in f:
            raise KeyError(f"Split '{split}' not in {h5_path}. Available: {list(f.keys())}")
        ds = f[split]
        m = min(n, len(ds))
        for i in range(m):
            out.append(decode_bytes_json(ds[i]))
    return out


def write_json_hdf5(out_path: Path, splits: dict[str, list[dict]]) -> None:
    """
    Write JSON-serializable records as variable-length bytes HDF5 datasets.
    Expected top-level keys in `splits`: 'train', 'val', 'test'.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vlen_bytes = h5py.special_dtype(vlen=bytes)
    with h5py.File(out_path, "w") as f:
        for split, records in splits.items():
            payloads = [encode_json_obj(rec) for rec in records]
            arr = np.array(payloads, dtype=object)
            f.create_dataset(split, data=arr, dtype=vlen_bytes, compression="gzip")
    print("Wrote:", out_path)


def inspect_hdf5(h5_path: Path, num: int = 3) -> None:
    """Print split names, counts, and a few decoded samples per split."""
    h5_path = h5_path.resolve()
    print("Inspecting:", h5_path)
    with h5py.File(h5_path, "r") as f:
        print("ROOT KEYS:", list(f.keys()))
        for split in ("train", "val", "test"):
            if split not in f:
                continue
            ds = f[split]
            print(f"\n=== {split} ===")
            print("count:", len(ds))
            show = min(num, len(ds))
            for i in range(show):
                rec = decode_bytes_json(ds[i])
                keys = list(rec.keys())
                print(f"- example[{i}] keys:", keys[:20])
                summary_keys = ("docid","title","label","label_str","mask_origin")
                summary = {k: rec.get(k) for k in summary_keys if k in rec}
                print("  summary:", summary)


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


# ==============================
#  Load data
# ==============================
def load_dataset(tokenizer, train_path, val_path=None, max_length=512, use_amr=False):
    train_ds = RebelAMRDataset(tokenizer, train_path, split="train", max_length=max_length, use_amr=use_amr)
    val_ds = None
    if val_path:
        val_ds = RebelAMRDataset(tokenizer, val_path, split="train", max_length=max_length, use_amr=use_amr)
    return train_ds, val_ds