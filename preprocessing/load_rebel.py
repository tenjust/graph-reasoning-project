import json
from pathlib import Path
from typing import Iterable

import h5py
from datasets import Dataset, DatasetDict

root_path = Path.cwd().parent


# Build (source, target) for seq2seq
def build_io(example):
    return {"source": example["context"], "target": linearize_triplets(example.get("triplets", []))}


def linearize_triplets(triplets):
    """Convert triplets to linearized string format with special tokens."""
    parts = []
    for t in triplets or []:
        if isinstance(t, dict):
            s = str(t.get("subject", "")).strip()
            r = str(t.get("relation", "")).strip()
            o = str(t.get("object", "")).strip()
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            s, r, o = (str(x).strip() for x in t[:3])
        else:
            continue

        if s and r and o:
            parts.extend(["<triplet>", s, "<subj>", r, "<obj>", o])

    return " ".join(parts)


def load_rebel_hdf5(path):
    """
    Loads REBEL dataset from HDF5 file where each element is a JSON string.
    Returns a DatasetDict with splits found in the file.
    """
    datasets = {}

    with h5py.File(path, "r") as f:
        split_names = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]

        if not split_names:
            raise RuntimeError("No split datasets found in HDF5.")

        for split in split_names:
            arr = f[split][()]
            rows = [json.loads(x.decode("utf-8") if isinstance(x, bytes) else x) for x in arr]

            contexts = [d["text"] for d in rows]
            triplets = [d["triplets"] for d in rows]

            datasets[split] = Dataset.from_dict({
                "context": contexts,
                "triplets": triplets
            })

    return DatasetDict(datasets)


def number_of_instances_hdf5(path: Path, split: str) -> int:
    """
    Get the number of instances in a given split of the REBEL dataset stored in HDF5.
    :param path: Path to HDF5 file.
    :param split: Split name ('train', 'validation', 'test').
    :return: Number of instances in the specified split.
    """
    if split not in ('train', 'validation', 'test'):
        raise ValueError(f"Invalid split name: {split}. Must be 'train', 'validation', or 'test'.")
    with h5py.File(path, "r") as f:
        if split not in f:
            raise KeyError(f"Split '{split}' not found in {path}. Available splits: {list(f.keys())}")
        dset = f[split]
        return dset.shape[0]


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


if __name__ == "__main__":
    path = root_path / "baselines" / "GraphLanguageModels" / "data" / "rebel_dataset" / "REBEL_AMR_TRIPLES.train.hdf5"

    split = "train"
    print(f"Total examples in split '{split}': {number_of_instances_hdf5(path, split)}")

    split_data = iterate_rebel_hdf5_split(path, split)
    cases_with_errors = 0
    ids_processed = []
    for i, instance in enumerate(split_data):
        if "AMR_based_triples" not in instance:
            cases_with_errors += 1
            continue
        ids_processed.append(i)
        # print(json.dumps(instance, indent=4, ensure_ascii=False))
    print(f"Processed {len(ids_processed)} examples, {cases_with_errors} had errors.")
    print("IDs processed:", ids_processed)
