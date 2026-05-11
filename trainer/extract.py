"""Extract the labelled subset of events into a single small parquet file.

Inner-joins event parquets with one or more label parquets on
`(customer_id, event_id)`. Default behaviour preserves the historical
`data_augmented/train_part_*.parquet` × `data/train_labels.parquet` flow;
production daily flow passes explicit globs of partitioned data instead.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

_DEFAULT_TRAIN_FILES = (
    "train_part_1.parquet",
    "train_part_2.parquet",
    "train_part_3.parquet",
)


def _resolve_event_paths(
    input_dir: Path | None,
    events_glob: str | Iterable[str] | None,
    train_files: list[str] | None,
) -> list[Path]:
    """Build the ordered list of event parquet paths from any of three modes."""
    if events_glob:
        patterns = [events_glob] if isinstance(events_glob, str) else list(events_glob)
        paths: list[Path] = []
        for pat in patterns:
            paths.extend(Path(p) for p in sorted(glob.glob(pat)))
        return paths
    if input_dir is None:
        raise ValueError("Either events_glob or input_dir must be provided.")
    names = list(train_files) if train_files else list(_DEFAULT_TRAIN_FILES)
    return [input_dir / n for n in names]


def _resolve_label_paths(
    labels_path: Path | None,
    labels_glob: str | Iterable[str] | None,
) -> list[Path]:
    if labels_glob:
        patterns = [labels_glob] if isinstance(labels_glob, str) else list(labels_glob)
        paths: list[Path] = []
        for pat in patterns:
            paths.extend(Path(p) for p in sorted(glob.glob(pat)))
        if not paths:
            raise FileNotFoundError(f"No label parquets matched {patterns!r}")
        return paths
    if labels_path is None:
        raise ValueError("Either labels_glob or labels_path must be provided.")
    return [labels_path]


def _load_labels(paths: list[Path], window_end: pd.Timestamp | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    has_dttm = False
    for p in paths:
        df = pq.read_table(str(p)).to_pandas()
        if "label_dttm" in df.columns:
            has_dttm = True
        frames.append(df)
    labels = pd.concat(frames, ignore_index=True)
    labels["customer_id"] = labels["customer_id"].astype("int64")
    labels["event_id"] = labels["event_id"].astype("int64")
    keep = ["customer_id", "event_id", "target"]
    if has_dttm and "label_dttm" in labels.columns:
        labels["label_dttm"] = pd.to_datetime(labels["label_dttm"], errors="coerce")
        if window_end is not None:
            cutoff_lo = window_end - pd.Timedelta(days=30)
            cutoff_hi = window_end - pd.Timedelta(days=7)
            in_window = labels["label_dttm"].between(cutoff_lo, cutoff_hi)
            labels = labels.loc[in_window].reset_index(drop=True)
    return labels[keep]


def extract_labelled_events(
    input_dir: Path | None = None,
    labels_path: Path | None = None,
    output_path: Path = Path("labelled_events.parquet"),
    train_files: list[str] | None = None,
    *,
    events_glob: str | Iterable[str] | None = None,
    labels_glob: str | Iterable[str] | None = None,
    label_window_end: pd.Timestamp | str | None = None,
    append: bool = False,
    batch_rows: int = 200_000,
    progress: bool = True,
) -> pd.DataFrame:
    """Extract labelled events from event partitions × labels.

    Behavior on 0 matches: returns empty DataFrame (does NOT raise). If
    `append=True` and `output_path` already exists, the new matches are
    unioned with the existing labelled set and deduplicated by
    `(customer_id, event_id)`. This makes daily reruns idempotent — no new
    labels today simply leaves the accumulated set intact.
    """
    event_paths = _resolve_event_paths(input_dir, events_glob, train_files)
    for p in event_paths:
        if not p.exists():
            raise FileNotFoundError(p)
    label_paths = _resolve_label_paths(labels_path, labels_glob)

    window_end = pd.to_datetime(label_window_end) if label_window_end else None
    labels = _load_labels(label_paths, window_end)
    if progress:
        print(f"[extract] {len(labels):,} labels loaded "
              f"({len(label_paths)} file(s), window_end={window_end})")

    chunks: list[pd.DataFrame] = []
    for path in event_paths:
        pf = pq.ParquetFile(str(path))
        total = pf.metadata.num_rows
        if progress:
            print(f"[extract] {path.name}: {total:,} rows")
        written = 0
        for batch in pf.iter_batches(batch_size=batch_rows):
            df = batch.to_pandas()
            written += len(df)
            merged = df.merge(labels, on=["customer_id", "event_id"], how="inner")
            if len(merged):
                chunks.append(merged)
            if progress:
                matched = sum(len(c) for c in chunks)
                print(f"  ... {written:,}/{total:,}  matched so far: {matched:,}", end="\r")
        if progress:
            print()

    new_labelled = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    if append and output_path.exists():
        existing = pq.read_table(str(output_path)).to_pandas()
        if progress:
            print(f"[extract] appending to existing {len(existing):,} rows")
        if len(new_labelled):
            combined = pd.concat([existing, new_labelled], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["customer_id", "event_id"], keep="last"
            ).reset_index(drop=True)
        else:
            combined = existing
        labelled = combined
    else:
        labelled = new_labelled

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(labelled):
        labelled.to_parquet(str(output_path), compression="snappy", index=False)
    else:
        # Write an empty parquet so downstream stages see the file.
        # Schema is whatever the first event parquet defines + a target column.
        empty_schema = pq.ParquetFile(str(event_paths[0])).schema_arrow
        import pyarrow as pa
        target_field = pa.field("target", pa.int64())
        empty_with_target = pa.schema(list(empty_schema) + [target_field])
        pa.parquet.write_table(empty_with_target.empty_table(), str(output_path))

    if progress:
        if len(labelled):
            print(f"[extract] wrote {len(labelled):,} labelled rows -> {output_path}")
            if "target" in labelled.columns:
                print(f"[extract] target dist: {labelled['target'].value_counts().to_dict()}")
        else:
            print(f"[extract] WARNING: no labelled events found "
                  f"(events={len(event_paths)} files, labels={len(labels):,}). "
                  f"Wrote empty {output_path}.")
    return labelled
