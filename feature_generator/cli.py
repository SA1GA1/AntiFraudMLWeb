"""CLI: augment all parquet files in a directory with 13 synthetic features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .augment import augment_file, discover_files, is_train_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="feature_generator",
        description="Augment parquet datasets with 13 synthetic fraud-detection features.",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data"),
        help="Directory containing source *.parquet files (default: ./data).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data_augmented"),
        help="Destination directory (default: ./data_augmented).",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to train_labels.parquet (default: <input>/train_labels.parquet).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for reproducible draws (default: 42).",
    )
    p.add_argument(
        "--batch-rows",
        type=int,
        default=200_000,
        help="Approx. rows per streaming batch (default: 200000).",
    )
    p.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Restrict to specific filenames (e.g. test.parquet train_part_1.parquet).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-batch progress output.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir: Path = args.input
    output_dir: Path = args.output

    if not input_dir.exists():
        print(f"error: input directory {input_dir} does not exist", file=sys.stderr)
        return 2

    labels_path = args.labels or (input_dir / "train_labels.parquet")
    if not labels_path.exists():
        print(f"warning: {labels_path} not found — train files will use risk-flag fallback only", file=sys.stderr)
        labels_path = None

    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(input_dir, args.files)
    if not files:
        print(f"error: no parquet files matched in {input_dir}", file=sys.stderr)
        return 2

    print(f"[cli] augmenting {len(files)} file(s) from {input_dir} -> {output_dir}")
    for src in files:
        dst = output_dir / src.name
        lbl = labels_path if is_train_file(src) else None
        augment_file(
            src,
            dst,
            labels_path=lbl,
            seed=args.seed,
            batch_rows=args.batch_rows,
            progress=not args.quiet,
        )
    print("[cli] all done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
