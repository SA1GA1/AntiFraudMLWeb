"""Row-group streaming augmentation of parquet files.

Reads one row group at a time, joins per-event `target` from the labels table
when available, calls `generate_features`, and writes the enriched batch out
through a single `ParquetWriter` so memory stays bounded for 600+ MB inputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .generators import NEW_COLUMNS, generate_features

# Approx rows per pyarrow batch; bounded by row-group size in the source file.
DEFAULT_BATCH_ROWS = 200_000


def _augment_batch(
    batch_df: pd.DataFrame,
    labels_lookup: Optional[dict],
    seed: int,
) -> pd.DataFrame:
    if labels_lookup is not None and len(batch_df) > 0:
        # Vectorised lookup via pandas merge on the small per-batch frame.
        target = batch_df.set_index(["customer_id", "event_id"]).index.map(
            labels_lookup.get
        )
        target = np.asarray(
            [np.nan if v is None else float(v) for v in target], dtype=np.float64
        )
    else:
        target = None

    new_cols = generate_features(batch_df, target, seed=seed)
    return pd.concat([batch_df.reset_index(drop=True), new_cols.reset_index(drop=True)], axis=1)


def _load_labels(labels_path: Path | None) -> Optional[dict]:
    if labels_path is None or not labels_path.exists():
        return None
    df = pq.read_table(str(labels_path)).to_pandas()
    return dict(zip(zip(df["customer_id"], df["event_id"]), df["target"]))


def augment_file(
    input_path: str | Path,
    output_path: str | Path,
    labels_path: Optional[str | Path] = None,
    seed: int = 42,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    progress: bool = True,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels_lookup = _load_labels(Path(labels_path)) if labels_path else None

    pf = pq.ParquetFile(str(input_path))
    total_rows = pf.metadata.num_rows
    schema_in = pf.schema_arrow

    if progress:
        print(
            f"[augment] {input_path.name}: {total_rows:,} rows, "
            f"{pf.num_row_groups} row group(s), labels={'yes' if labels_lookup else 'no'}"
        )

    writer: Optional[pq.ParquetWriter] = None
    written = 0
    try:
        for record_batch in pf.iter_batches(batch_size=batch_rows):
            df = record_batch.to_pandas()
            aug = _augment_batch(df, labels_lookup, seed=seed)
            table = pa.Table.from_pandas(aug, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_path),
                    table.schema,
                    compression="zstd",
                    compression_level=3,
                )
            writer.write_table(table)
            written += len(df)
            if progress:
                pct = 100.0 * written / max(total_rows, 1)
                print(f"  ... wrote {written:,}/{total_rows:,} rows ({pct:5.1f}%)", end="\r")
    finally:
        if writer is not None:
            writer.close()

    if progress:
        print()
        print(f"[augment] done -> {output_path} ({written:,} rows)")


def discover_files(input_dir: Path, only: Optional[list[str]] = None) -> list[Path]:
    files = sorted(
        p
        for p in input_dir.glob("*.parquet")
        if p.name != "train_labels.parquet" and not p.name.endswith("_aug.parquet")
    )
    if only is None:
        return files
    name_set = set(only)
    matched = [p for p in files if p.name in name_set or p.stem in name_set]
    missing = name_set - {p.name for p in matched} - {p.stem for p in matched}
    if missing:
        raise FileNotFoundError(f"Files not found in {input_dir}: {sorted(missing)}")
    return matched


def is_train_file(path: Path) -> bool:
    return path.name.startswith("train_part_")
