import json
from pathlib import Path
import h5py
from datasets import Dataset, DatasetDict

root_path = Path.cwd().parent

# def _decode(b):
#     return b.decode("utf-8", "ignore") if isinstance(b, (bytes, bytearray)) else b

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

def iterate_rebel_hdf5_split(path, split, batch_size=1000):
    with h5py.File(Path(path), "r") as f:
        dset = f[split]
        total = dset.shape[0]
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = dset[start:end]
            for x in batch:
                yield json.loads(x.decode("utf-8", "ignore"))
            break


# Build (source, target) for seq2seq
def build_io(example):
    return {"source": example["context"], "target": linearize_triplets(example.get("triplets", []))}

if __name__ == "__main__":
    path = root_path / "graph-reasoning-project" / "baselines" / "GraphLanguageModels" / "data" / "rebel_dataset" / "rebel.hdf5"
    ds = load_rebel_hdf5(path)

    print("Splits found:", list(ds.keys()))
    some_split = list(ds.keys())[0] # Splits found: ['test', 'train', 'validation']

    # {'context': 'Coburg Peak (, ) is the rocky peak rising to 783 m in Erul Heights on Trinity Peninsula in Graham Land, Antarctica.
    # It is surmounting Cugnot Ice Piedmont to the northeast.\n\nThe peak is named after the Bulgarian royal house of Coburg (Saxe-Coburg-Gotha), 1887–1946.',
    # 'triplets': [['Graham Land', 'continent', 'Antarctica'], ['Trinity Peninsula', 'continent', 'Antarctica'], ['Coburg Peak', 'continent', 'Antarctica'],
    # ['Cugnot Ice Piedmont', 'continent', 'Antarctica'], ['Erul Heights', 'continent', 'Antarctica'], ['Trinity Peninsula', '<mask>', 'Graham Land']]}
    for instance in ds[some_split][:5]:
        print(json.dumps(instance, indent=4))