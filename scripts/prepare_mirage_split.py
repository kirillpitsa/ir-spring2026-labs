#!/opt/homebrew/Caskroom/miniforge/base/envs/ir_env/bin/python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download MIRAGE if needed and create a grouped, source-stratified train/test split.",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    from src import prepare_local_mirage_split

    metadata = prepare_local_mirage_split(
        data_root=args.data_root,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
