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
    batch_rows: int = 200_000,
    progress: bool = True,
) -> pd.DataFrame:
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

    if not chunks:
        raise RuntimeError("No labelled events found in the configured event paths.")

    labelled = pd.concat(chunks, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(str(output_path), compression="snappy", index=False)
    if progress:
        print(f"[extract] wrote {len(labelled):,} labelled rows -> {output_path}")
        print(f"[extract] target dist: {labelled['target'].value_counts().to_dict()}")
    return labelled
