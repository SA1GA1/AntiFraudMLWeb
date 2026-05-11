"""Extract the labelled subset of events into a single small parquet file.

Inner-joins each `data_augmented/train_part_*.parquet` with
`data/train_labels.parquet` on (customer_id, event_id). Result is ~87 514
rows × (36 + 1 target) columns, fitting easily in memory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def extract_labelled_events(
    input_dir: Path,
    labels_path: Path,
    output_path: Path,
    train_files: list[str] | None = None,
    batch_rows: int = 200_000,
    progress: bool = True,
) -> pd.DataFrame:
    if train_files is None:
        train_files = [
            "train_part_1.parquet",
            "train_part_2.parquet",
            "train_part_3.parquet",
        ]

    labels = pq.read_table(str(labels_path)).to_pandas()
    labels["customer_id"] = labels["customer_id"].astype("int64")
    labels["event_id"] = labels["event_id"].astype("int64")
    labels = labels[["customer_id", "event_id", "target"]]

    if progress:
        print(f"[extract] {len(labels):,} labels loaded")

    chunks: list[pd.DataFrame] = []
    for fname in train_files:
        path = input_dir / fname
        if not path.exists():
            raise FileNotFoundError(path)
        pf = pq.ParquetFile(str(path))
        total = pf.metadata.num_rows
        if progress:
            print(f"[extract] {fname}: {total:,} rows")
        written = 0
        for batch in pf.iter_batches(batch_size=batch_rows):
            df = batch.to_pandas()
            written += len(df)
            merged = df.merge(labels, on=["customer_id", "event_id"], how="inner")
            if len(merged):
                chunks.append(merged)
            if progress:
                print(f"  ... {written:,}/{total:,}  matched so far: {sum(len(c) for c in chunks):,}", end="\r")
        if progress:
            print()

    if not chunks:
        raise RuntimeError("No labelled events found in train_part_* files.")

    labelled = pd.concat(chunks, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(str(output_path), compression="snappy", index=False)
    if progress:
        print(f"[extract] wrote {len(labelled):,} labelled rows -> {output_path}")
        print(f"[extract] target dist: {labelled['target'].value_counts().to_dict()}")
    return labelled
