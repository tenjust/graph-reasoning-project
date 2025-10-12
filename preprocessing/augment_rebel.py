from __future__ import annotations
from pathlib import Path
import argparse
import sys

# Add a project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# External dependencies / local modules
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


def add_amr_to_records(records: list[dict], limit: int = None) -> list[dict]:
    """Add 'AMR_based_triples' field to each record."""
    end = len(records) if limit is None else min(limit, len(records))

    for i in range(end):
        text = records[i].get("text", "")
        records[i]["AMR_based_triples"] = get_amr_triples(text) if text else []

    return records


# ------------------------ CLI ------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Process REBEL dataset and optionally augment with AMR triples")
    p.add_argument("--src", type=str, default=str(REBEL_DIR / "rebel.hdf5"),
                   help="Source HDF5 file to read from (default to REBEL_DIR/rebel.hdf5)")
    p.add_argument("--out", type=str, default=str(REBEL_DIR / "REBEL_AMR_TRIPLES.hdf5"),
                   help="Output HDF5 file to write to")
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if args.inspect:
        inspect_hdf5(src, num=3)
        return

    print("Loading subsets from:", src)
    train = load_original_split(src, "train", n=args.n_train)
    val   = load_original_split(src, "val",   n=args.n_val)
    test  = load_original_split(src, "test",  n=args.n_test)

    if args.amr:
        print("Adding AMR triples...")
        add_amr_to_records(train)
        add_amr_to_records(val)
        add_amr_to_records(test)

    # Write a result (original or augmented) to the output HDF5
    write_json_hdf5(out, {"train": train, "val": val, "test": test})


if __name__ == "__main__":
    main()