import json
from pathlib import Path
import h5py
from datasets import Dataset, DatasetDict
import os
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

root_path = Path(os.getcwd()).parent

SPECIAL_TOKENS = ["<triplet>", "<subj>", "<obj>"]

def _decode(b):
    return b.decode("utf-8", "ignore") if isinstance(b, (bytes, bytearray)) else b

def _pick_text(d):
    # Moritz’s JSON rows usually have 'text'. Fall back to common alternatives.
    for k in ("text", "context", "sentence", "source"):
        if k in d and d[k]:
            return d[k]
    return ""

def _pick_triplets(d):
    # Usually 'triplets'. Be tolerant to variants.
    for k in ("triplets", "relations", "labels", "targets", "target"):
        if k in d and d[k] is not None:
            return d[k]
    return []

def linearize_triplets(triplets):
    parts = []
    for t in triplets or []:
        if isinstance(t, dict):
            s = str(t.get("subject","")).strip()
            r = str(t.get("relation","")).strip()
            o = str(t.get("object","")).strip()
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            s, r, o = map(lambda x: str(x).strip(), t[:3])
        else:
            continue
        if s and r and o:
            parts += ["<triplet>", s, "<subj>", r, "<obj>", o]
    return " ".join(parts)


def load_rebel_hdf5(path):
    """
    Loads an HDF5 created by Moritz's script:
      - datasets at the root named by split ('train', 'val', 'test', etc.)
      - each element is a JSON string with at least 'text' and 'triplets' (or variants)
    Returns a DatasetDict with splits as found in the file.
    """
    dd = {}
    with h5py.File(Path(path), "r") as f:
        # auto-detect split names that are datasets (not groups)
        split_names = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        if not split_names:
            raise RuntimeError("No split datasets found in HDF5 (expected datasets like 'train', 'val', 'test').")

        # normalize aliases (val/dev -> validation)
        alias = {"val":"validation", "valid":"validation", "dev":"validation"}
        for split in split_names:
            canon = alias.get(split.lower(), split.lower())
            arr = f[split][()]  # array of bytes/strings
            rows = [json.loads(_decode(x)) for x in arr]

            contexts = [_pick_text(d) for d in rows]
            triplets = [_pick_triplets(d) for d in rows]

            dd[canon] = Dataset.from_dict({"context": contexts, "triplets": triplets})

    return DatasetDict(dd)


# Build (source, target) for seq2seq
def build_io(example):
    return {"source": example["context"], "target": linearize_triplets(example.get("triplets", []))}

path = os.path.join(root_path, "graph-reasoning-project", "baselines", "GraphLanguageModels","data", "rebel_dataset", "rebel.hdf5")
ds = load_rebel_hdf5(path)

print("Splits found:", list(ds.keys()))
# Splits found: ['test', 'train', 'validation']
some_split = list(ds.keys())[0]
print(ds[some_split][0])  # peek first example
# {'context': 'Coburg Peak (, ) is the rocky peak rising to 783 m in Erul Heights on Trinity Peninsula in Graham Land, Antarctica. It is surmounting Cugnot Ice Piedmont to the northeast.\n\nThe peak is named after the Bulgarian royal house of Coburg (Saxe-Coburg-Gotha), 1887–1946.', 'triplets': [['Graham Land', 'continent', 'Antarctica'], ['Trinity Peninsula', 'continent', 'Antarctica'], ['Coburg Peak', 'continent', 'Antarctica'], ['Cugnot Ice Piedmont', 'continent', 'Antarctica'], ['Erul Heights', 'continent', 'Antarctica'], ['Trinity Peninsula', '<mask>', 'Graham Land']]}

# ds_io = ds.map(build_io, remove_columns=[c for c in ds[next(iter(ds))].column_names if c not in ("context","triplets")])
#
# # Optional: drop empty targets
# ds_io = ds_io.filter(lambda e: bool(e["target"].strip()))
