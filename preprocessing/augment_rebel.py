# Description: Incrementally process REBEL dataset to add AMR-based triples, with error handling and resumability.
#              Each split can be processed separately, allowing parallel execution.
#              Outputs are stored in HDF5 format, compatible with existing REBEL loaders.
#              Example usage at the end of the file.
# ---------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import argparse
import sys
import json
import traceback
import h5py
from tqdm import tqdm

# Add a project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# External deps / local modules
from amr_graph import AMRGraph
from config import REBEL_DIR
from preprocessing.hdf5_utils import (
    load_original_split, write_json_hdf5, inspect_hdf5
)

# ---------- AMR augmentation  ----------

def get_amr_triples(text: str) -> list[list[str]]:
    """Extract AMR triples from text."""
    graph = AMRGraph(text)
    return [[str(x) for x in t.after] for t in graph.triples if t.after]


def add_amr_to_record(record: dict) -> dict:
    """Add 'AMR_based_triples' field to a single record (in-place)."""
    text = record.get("text", "")
    record["AMR_based_triplets"] = get_amr_triples(text) if text else []
    record["_amr_error"] = ""
    return record

def add_empty_record(record: dict, error: Exception) -> dict:
    """Add an empty AMR result to a single record in case of errors (in-place)."""
    record.setdefault("AMR_based_triplets", [])
    record["_amr_error"] = str(error).strip()
    return record

# ------------------------ Incremental HDF5 helpers ------------------------ #

def _ensure_split_dataset(h5f: h5py.File, split: str) -> h5py.Dataset:
    """
    Ensure an appendable string dataset exists at the root named exactly '/<split>'.
    Each element is a JSON row (UTF-8). This matches the other reader.
    """
    node = h5f.get(split)
    if node:
        if isinstance(node, h5py.Dataset):
            return node
        else:
            raise TypeError(
                f"Expected '/{split}' to be a dataset, found {type(node)}. "
                f"Delete or rename this group (it probably contains 'data')."
            )
    # if not found, create it
    dset = h5f.create_dataset(
        split,
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype('utf-8'),
        chunks=True
    )
    dset.attrs["written"] = 0
    return dset


def _written_count(h5f: h5py.File, split: str) -> int:
    """Return how many records have been written to the given split dataset."""
    if split in h5f and isinstance(h5f[split], h5py.Dataset):
        return int(h5f[split].attrs.get("written", 0))
    return 0


def _append_record(h5f: h5py.File, split: str, obj: dict):
    """Append a single JSON-serializable object to the given split dataset."""
    dset = _ensure_split_dataset(h5f, split)
    n = dset.shape[0]
    dset.resize((n + 1,))
    dset[n] = json.dumps(obj, ensure_ascii=False)  # for proper printing: dset[n].decode("utf-8")
    dset.attrs["written"] = int(dset.attrs.get("written", 0)) + 1


# ------------------------ CLI ------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Process REBEL dataset and optionally augment with AMR triples (incremental & resumable)")
    p.add_argument("--src", type=str, default=str(REBEL_DIR / "rebel.hdf5"),
                   help="Source HDF5 file to read from (default: REBEL_DIR/rebel.hdf5)")
    p.add_argument("--out", type=str, default=None,
                   help="Output HDF5 file. If not set, defaults to REBEL_AMR_TRIPLES.<split>.hdf5")
    p.add_argument("--n-train", type=int, default=100000,
                   help="Number of training examples to process")
    p.add_argument("--n-val", type=int, default=10000,
                   help="Number of validation examples to process")
    p.add_argument("--n-test", type=int, default=30000,
                   help="Number of test examples to process")
    p.add_argument("--inspect", action="store_true",
                   help="Inspect the source HDF5 file and exit")
    p.add_argument("--resume", action="store_true",
                   help="Resume from where the output file last left off (per split).")
    p.add_argument("--flush-every", type=int, default=50,
                   help="Flush HDF5 to disk every K records (default: 50). Use 1 for fully per-instance.")
    p.add_argument("--skip-errors", action="store_true",
                   help="If set, logs and skips records that raise exceptions (keeps going).")
    p.add_argument("--split", choices=["train", "val", "test"], required=True,
                   help="Which split to process in this run.")
    return p.parse_args()


# ------------------------ Processing ------------------------ #

def process_split(src: Path, out: Path, split: str, n: int,
                  resume: bool, flush_every: int, skip_errors: bool) -> None:
    """
    Load up to n examples from `split`, process incrementally, append to `out` HDF5.
    If `resume` is True, skip records already present in `out` for that split.
    :param src: Source HDF5 file
    :param out: Output HDF5 file (created if missing)
    :param split: One of "train", "val", "test"
    :param n: Number of records to process (max)
    :param resume: If True, skip records already present in `out`
    :param flush_every: Flush to disk every K records
    :param skip_errors: If True, log and skip records that raise exceptions
    :return: None
    """
    print(f"\n[Split: {split}] Loading up to {n} examples from: {src}")
    records = load_original_split(src, split, n=n)
    total = len(records)

    # open output in append mode
    with h5py.File(out, "a") as h5f:
        # Determine how many we've already written
        start_idx = _written_count(h5f, split) if resume else 0
        start_idx = min(start_idx, total)
        if start_idx > 0:
            print(f"[Split: {split}] Resuming at index {start_idx}/{total}")

        # Iterate and append
        since_last_flush = 0
        for i in tqdm(range(start_idx, total), desc="Processing data instances.."):
            rec = records[i]

            try:
                add_amr_to_record(rec)
            except Exception as e:
                if skip_errors:
                    print(
                        f"[Split: {split}] [idx {i}] "
                        f"Skipping error occurred when creating AMR data: {e}"
                    )
                    traceback.print_exc(limit=1)
                    add_empty_record(rec, e)
                else:
                    raise e

            _append_record(h5f, split, rec)
            since_last_flush += 1

            # Periodic flush
            if since_last_flush >= flush_every:
                if flush_every > 1:
                    print(f"[Split: {split}] [{start_idx + i + 1}/{total}] Saving {since_last_flush} last instances..")
                h5f.flush()
                since_last_flush = 0

        # final flush
        h5f.flush()

    # Explicitly free the big list before moving to the next split
    del records
    print(f"[Split: {split}] Done.")


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    if args.out is None:
        out = (REBEL_DIR / f"REBEL_AMR_TRIPLES.{args.split}.hdf5").expanduser().resolve()
    else:
        out = Path(args.out).expanduser().resolve()

    if args.inspect:
        inspect_hdf5(src, num=3)
        return

    print("Source:", src)
    print("Output:", out)

    # Process only the requested split (safe for parallel runs)
    n_map = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    n = n_map[args.split]
    process_split(src, out, args.split, n, args.resume, args.flush_every, args.skip_errors)

    print(f"\nDone. Split {args.split} written to {out}. Re-run with --resume to continue if interrupted.")


if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------
# Example usage:
# Process each split separately (safe for parallel execution):
#
# python3 augment_rebel.py --split train --flush-every 25 --skip-errors --resume
# python3 augment_rebel.py --split val   --flush-every 25 --skip-errors --resume
# python3 augment_rebel.py --split test  --flush-every 25 --skip-errors --resume
#
# You can run them in parallel, e.g.:
#   parallel -j3 --lb \
#     "python augment_rebel.py --split {} --flush-every 25 --skip-errors --resume" ::: train val test
#
# After all runs complete, merge outputs using merge_hdf5_splits.py (if desired):
#   python3 merge_hdf5_splits.py REBEL_AMR_TRIPLES.all.hdf5 \
#     REBEL_AMR_TRIPLES.train.hdf5 REBEL_AMR_TRIPLES.val.hdf5 REBEL_AMR_TRIPLES.test.hdf5
# ---------------------------------------------------------------------------