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
    record["AMR_based_triples"] = get_amr_triples(text) if text else []
    return record


# ------------------------ Incremental HDF5 helpers ------------------------ #

def _ensure_split_dataset(h5f: h5py.File, split: str) -> h5py.Dataset:
    """
    Ensure an appendable string dataset exists at the root named exactly '/<split>'.
    Each element is a JSON row (UTF-8). This matches the other reader.
    """
    node = h5f.get(split)
    if node is not None:
        if isinstance(node, h5py.Dataset):
            return node
        else:
            raise TypeError(
                f"Expected '/{split}' to be a dataset, found {type(node)}. "
                f"Delete or rename this group (it probably contains 'data')."
            )
    str_dt = h5py.string_dtype()
    dset = h5f.create_dataset(
        split,
        shape=(0,),
        maxshape=(None,),
        dtype=str_dt,
        chunks=True
    )
    dset.attrs["written"] = 0
    return dset


def _written_count(h5f: h5py.File, split: str) -> int:
    if split in h5f and isinstance(h5f[split], h5py.Dataset):
        return int(h5f[split].attrs.get("written", 0))
    return 0


def _append_record(h5f: h5py.File, split: str, obj: dict):
    dset = _ensure_split_dataset(h5f, split)
    n = dset.shape[0]
    dset.resize((n + 1,))
    dset[n] = json.dumps(obj, ensure_ascii=False).encode("utf-8")
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
    p.add_argument("--amr", action="store_true",
                   help="Augment records with AMR-based triples")
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

def process_split(src: Path, out: Path, split: str, n: int, use_amr: bool,
                  resume: bool, flush_every: int, skip_errors: bool):
    """
    Load up to n examples from `split`, process incrementally, append to `out` HDF5.
    If `resume` is True, skip records already present in `out` for that split.
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
        for i in tqdm(range(start_idx, total), desc="Processing sentence.."):
            rec = records[i]

            if use_amr:
                try:
                    add_amr_to_record(rec)
                    rec["triplets"] = rec.get("AMR_based_triples", [])
                except Exception as e:
                    if skip_errors:
                        print(f"[Split: {split}] [idx {i}] AMR error: {e}")
                        traceback.print_exc(limit=1)
                        rec["_amr_error"] = str(e)
                        rec["_amr_failed"] = True
                        rec.setdefault("triplets", [])
                    else:
                        raise
            else:
                rec.setdefault("triplets", [])

            _append_record(h5f, split, rec)
            since_last_flush += 1

            # Periodic flush
            if since_last_flush >= flush_every:
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
    process_split(src, out, args.split, n, args.amr, args.resume, args.flush_every, args.skip_errors)

    print(f"\nDone. Split {args.split} written to {out}. Re-run with --resume to continue if interrupted.")


if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------
# Example usage:
# Process each split separately (safe for parallel execution):
#
# python augment_rebel.py --split train --amr --flush-every 25 --skip-errors --resume
# python augment_rebel.py --split val   --amr --flush-every 25 --skip-errors --resume
# python augment_rebel.py --split test  --amr --flush-every 25 --skip-errors --resume
#
# You can run them in parallel, e.g.:
#   parallel -j3 --lb \
#     "python augment_rebel.py --split {} --amr --flush-every 25 --skip-errors --resume" ::: train val test
#
# After all runs complete, merge outputs using merge_hdf5_splits.py (if desired):
#   python merge_hdf5_splits.py REBEL_AMR_TRIPLES.all.hdf5 \
#     REBEL_AMR_TRIPLES.train.hdf5 REBEL_AMR_TRIPLES.val.hdf5 REBEL_AMR_TRIPLES.test.hdf5
# ---------------------------------------------------------------------------