"""Потоковая агрегация клиентских признаков из data_augmented/.

Все колонки соответствуют web-fraud схеме task.md (71 признак).

Поддерживает два режима:
- **full** (по умолчанию) — пересчёт по всем переданным файлам с нуля.
- **incremental** — догружает persisted state (raw sums + extremes) и
  применяет только новые партиции, сохраняя обновлённый state. Это нужно
  для daily flow, где `events/dt=YYYY-MM-DD/` пополняется без полного
  rescan'а 108M строк.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# task.md бинарные колонки → доли (mean).
_BINARY_COLS = [
    "is_developer_tools", "is_headless_browser",
    "is_incognito",
    "is_vpn_detected", "is_proxy_detected", "is_tor_detected",
    "is_new_device", "is_new_browser",
]

# task.md числовые колонки → mean.
_MEAN_COLS = [
    "network_rtt_avg_ms",
    "mouse_velocity_avg", "mouse_acceleration_avg",
    "mouse_jitter_score", "mouse_linearity_score",
    "click_duration_avg_ms", "right_click_count", "scroll_velocity_avg",
    "keyboard_typing_speed_median_ms", "keyboard_typing_speed_std_dev",
    "keyboard_typing_rhythm_cv",
    "backspace_ratio", "clipboard_paste_ratio",
    "copy_events_count", "paste_events_count",
    "tab_switch_count", "focus_blur_count",
    "form_fill_duration_sec", "idle_time_before_submit_sec",
    "error_correction_ratio", "hover_time_avg_ms",
    "double_click_count", "drag_drop_events",
    "resize_events_count", "zoom_level",
    "session_duration_sec", "pages_visited_count",
    "installed_fonts_count",
    "failed_login_attempts", "time_since_last_login_sec",
    "device_trust_score",
]

FEATURE_COLUMNS = [
    # Amount
    "event_count", "amt_mean", "amt_std", "amt_max", "amt_log_mean",
    # Binary shares
    "dev_tools_share", "headless_share", "incognito_share",
    "vpn_share", "proxy_share", "tor_share",
    "new_device_share", "new_browser_share",
    # Network
    "mean_rtt",
    # Mouse biometrics
    "mean_mouse_velocity", "mean_mouse_accel",
    "mean_mouse_jitter", "mean_mouse_linearity",
    # Click / scroll
    "mean_click_duration", "mean_right_clicks", "mean_scroll_velocity",
    # Keyboard
    "mean_typing_median_ms", "mean_typing_std", "mean_typing_cv",
    # Clipboard / form
    "mean_backspace", "mean_clipboard_paste",
    "mean_copy_events", "mean_paste_events",
    "mean_tab_switch", "mean_focus_blur",
    "mean_form_fill", "mean_idle_before_submit",
    "mean_error_correction", "mean_hover_time",
    "mean_double_click", "mean_drag_drop",
    "mean_resize_events", "mean_zoom_level",
    # Session shape
    "mean_session_duration", "mean_pages_visited",
    # Fonts
    "mean_installed_fonts",
    # Login / trust
    "mean_failed_logins", "mean_time_since_login", "mean_device_trust",
    # Categorical-derived shares
    "foreign_isp_share", "mobile_os_share",
    # Temporal
    "hours_span", "night_ops_share", "weekend_share",
]

_READ_COLS = [
    "customer_id", "event_dttm", "operaton_amt",
    "os_type", "isp_name",
] + _BINARY_COLS + _MEAN_COLS

# Иностранные ISP — те же что в feature_generator (для согласованности).
_FOREIGN_ISP_SET = {
    "Vodafone DE", "T-Mobile DE", "Orange FR", "China Mobile", "China Telecom",
    "AT&T", "Comcast", "Kcell", "Beltelecom", "Turk Telekom",
}
_MOBILE_OS_SET = {"Android", "iOS"}


def _prepare_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    amt = df["operaton_amt"].astype(np.float64).fillna(0.0)
    df["operaton_amt"] = amt
    df["_amt_sq"] = amt * amt
    df["_amt_log"] = np.log1p(amt.clip(lower=0.0))

    for col in _BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.float32).clip(0, 1)
        else:
            df[col] = np.float32(0.0)

    for col in _MEAN_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32).fillna(0.0)
        else:
            df[col] = np.float32(0.0)

    # Categorical-derived shares.
    if "isp_name" in df.columns:
        df["_foreign_isp"] = df["isp_name"].fillna("").isin(_FOREIGN_ISP_SET).astype(np.float32)
    else:
        df["_foreign_isp"] = np.float32(0.0)
    if "os_type" in df.columns:
        df["_mobile_os"] = df["os_type"].fillna("").isin(_MOBILE_OS_SET).astype(np.float32)
    else:
        df["_mobile_os"] = np.float32(0.0)

    # Temporal.
    dt = pd.to_datetime(df["event_dttm"], errors="coerce")
    hour = dt.dt.hour.fillna(12).astype(np.int16)
    dow = dt.dt.dayofweek.fillna(0).astype(np.int16)
    df["_is_night"] = ((hour >= 0) & (hour < 6)).astype(np.float32)
    df["_is_weekend"] = (dow >= 5).astype(np.float32)
    valid = dt.notna().to_numpy()
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    sec_dt = dt.astype("datetime64[s]", copy=False).to_numpy().view("int64")
    df["_ts"] = np.where(valid, sec_dt, np.nan).astype(np.float64)
    return df


_SUM_COLS_FROM_BATCH = (
    ["operaton_amt", "_amt_sq", "_amt_log"]
    + _BINARY_COLS
    + _MEAN_COLS
    + ["_foreign_isp", "_mobile_os", "_is_night", "_is_weekend"]
)


_STATE_EXTRA_COLS = ("__max_amt", "__min_ts", "__max_ts")


def _processed_files_path(state_path: Path) -> Path:
    """Companion parquet next to state — records which input files were already
    folded into the raw sums. Same dir, same stem, `.files.parquet` suffix.
    """
    return state_path.with_name(state_path.stem + ".files.parquet")


class CustomerAggregator:
    def __init__(self) -> None:
        self._sums: pd.DataFrame | None = None
        self._max_amt: pd.Series | None = None
        self._min_ts: pd.Series | None = None
        self._max_ts: pd.Series | None = None
        # File-level idempotency: (path, size, mtime_ns) of every parquet
        # whose contents are already in the raw sums.
        self._processed_files: set[tuple[str, int, int]] = set()

    # ------------------------------------------------------------------
    # Persistence — for incremental daily runs.
    # ------------------------------------------------------------------

    @staticmethod
    def _file_key(path: Path) -> tuple[str, int, int]:
        st = path.stat()
        return (str(path), int(st.st_size), int(st.st_mtime_ns))

    def should_process(self, path: Path) -> bool:
        """True if `path` is not yet in the processed-files set."""
        return self._file_key(path) not in self._processed_files

    def mark_processed(self, path: Path) -> None:
        self._processed_files.add(self._file_key(path))

    def save_state(self, path: Path) -> None:
        if self._sums is None:
            raise RuntimeError("Nothing to save — aggregator received no batches.")
        state = self._sums.copy()
        state["__max_amt"] = self._max_amt.reindex(state.index)
        state["__min_ts"] = self._min_ts.reindex(state.index)
        state["__max_ts"] = self._max_ts.reindex(state.index)
        path.parent.mkdir(parents=True, exist_ok=True)
        state.reset_index().to_parquet(str(path), compression="snappy", index=False)

        # Persist the processed-files set alongside the state.
        files_df = pd.DataFrame(
            list(self._processed_files), columns=["path", "size", "mtime_ns"]
        )
        files_df.to_parquet(str(_processed_files_path(path)),
                            compression="snappy", index=False)

    @classmethod
    def load_state(cls, path: Path) -> "CustomerAggregator":
        agg = cls()
        if not path.exists():
            return agg
        df = pq.read_table(str(path)).to_pandas().set_index("customer_id")
        agg._max_amt = df["__max_amt"]
        agg._min_ts = df["__min_ts"]
        agg._max_ts = df["__max_ts"]
        agg._sums = df.drop(columns=list(_STATE_EXTRA_COLS))

        files_path = _processed_files_path(path)
        if files_path.exists():
            files_df = pq.read_table(str(files_path)).to_pandas()
            agg._processed_files = {
                (str(r.path), int(r.size), int(r.mtime_ns))
                for r in files_df.itertuples(index=False)
            }
        return agg

    def process(self, df: pd.DataFrame) -> None:
        df = _prepare_batch(df)
        gb = df.groupby("customer_id", sort=False)
        part_sums = gb[_SUM_COLS_FROM_BATCH].sum()
        part_sums["__count"] = gb.size()
        part_max_amt = gb["operaton_amt"].max()
        ts = df.dropna(subset=["_ts"])
        if len(ts):
            ts_gb = ts.groupby("customer_id", sort=False)["_ts"]
            part_min_ts = ts_gb.min()
            part_max_ts = ts_gb.max()
        else:
            part_min_ts = pd.Series(dtype="float64")
            part_max_ts = pd.Series(dtype="float64")

        if self._sums is None:
            self._sums = part_sums
            self._max_amt = part_max_amt
            self._min_ts = part_min_ts
            self._max_ts = part_max_ts
        else:
            self._sums = self._sums.add(part_sums, fill_value=0.0)
            self._max_amt = pd.concat([self._max_amt, part_max_amt]).groupby(level=0).max()
            self._min_ts = pd.concat([self._min_ts, part_min_ts]).groupby(level=0).min()
            self._max_ts = pd.concat([self._max_ts, part_max_ts]).groupby(level=0).max()

    def finalize(self) -> pd.DataFrame:
        if self._sums is None:
            raise RuntimeError("Aggregator received no batches.")
        n = self._sums["__count"].astype(np.float64).clip(lower=1.0)
        out = pd.DataFrame(index=self._sums.index)
        out["event_count"] = self._sums["__count"].astype(np.int64)
        out["amt_mean"] = self._sums["operaton_amt"] / n
        var = (self._sums["_amt_sq"] / n) - (out["amt_mean"] ** 2)
        out["amt_std"] = np.sqrt(np.clip(var, 0.0, None))
        out["amt_max"] = self._max_amt.reindex(out.index).fillna(0.0)
        out["amt_log_mean"] = self._sums["_amt_log"] / n

        # Binary shares.
        out["dev_tools_share"] = self._sums["is_developer_tools"] / n
        out["headless_share"] = self._sums["is_headless_browser"] / n
        out["incognito_share"] = self._sums["is_incognito"] / n
        out["vpn_share"] = self._sums["is_vpn_detected"] / n
        out["proxy_share"] = self._sums["is_proxy_detected"] / n
        out["tor_share"] = self._sums["is_tor_detected"] / n
        out["new_device_share"] = self._sums["is_new_device"] / n
        out["new_browser_share"] = self._sums["is_new_browser"] / n

        # Network.
        out["mean_rtt"] = self._sums["network_rtt_avg_ms"] / n

        # Mouse.
        out["mean_mouse_velocity"] = self._sums["mouse_velocity_avg"] / n
        out["mean_mouse_accel"] = self._sums["mouse_acceleration_avg"] / n
        out["mean_mouse_jitter"] = self._sums["mouse_jitter_score"] / n
        out["mean_mouse_linearity"] = self._sums["mouse_linearity_score"] / n

        # Click / scroll.
        out["mean_click_duration"] = self._sums["click_duration_avg_ms"] / n
        out["mean_right_clicks"] = self._sums["right_click_count"] / n
        out["mean_scroll_velocity"] = self._sums["scroll_velocity_avg"] / n

        # Keyboard.
        out["mean_typing_median_ms"] = self._sums["keyboard_typing_speed_median_ms"] / n
        out["mean_typing_std"] = self._sums["keyboard_typing_speed_std_dev"] / n
        out["mean_typing_cv"] = self._sums["keyboard_typing_rhythm_cv"] / n

        # Clipboard / form interactions.
        out["mean_backspace"] = self._sums["backspace_ratio"] / n
        out["mean_clipboard_paste"] = self._sums["clipboard_paste_ratio"] / n
        out["mean_copy_events"] = self._sums["copy_events_count"] / n
        out["mean_paste_events"] = self._sums["paste_events_count"] / n
        out["mean_tab_switch"] = self._sums["tab_switch_count"] / n
        out["mean_focus_blur"] = self._sums["focus_blur_count"] / n
        out["mean_form_fill"] = self._sums["form_fill_duration_sec"] / n
        out["mean_idle_before_submit"] = self._sums["idle_time_before_submit_sec"] / n
        out["mean_error_correction"] = self._sums["error_correction_ratio"] / n
        out["mean_hover_time"] = self._sums["hover_time_avg_ms"] / n
        out["mean_double_click"] = self._sums["double_click_count"] / n
        out["mean_drag_drop"] = self._sums["drag_drop_events"] / n
        out["mean_resize_events"] = self._sums["resize_events_count"] / n
        out["mean_zoom_level"] = self._sums["zoom_level"] / n

        # Session shape.
        out["mean_session_duration"] = self._sums["session_duration_sec"] / n
        out["mean_pages_visited"] = self._sums["pages_visited_count"] / n

        # Fonts.
        out["mean_installed_fonts"] = self._sums["installed_fonts_count"] / n

        # Login / trust.
        out["mean_failed_logins"] = self._sums["failed_login_attempts"] / n
        out["mean_time_since_login"] = self._sums["time_since_last_login_sec"] / n
        out["mean_device_trust"] = self._sums["device_trust_score"] / n

        # Categorical-derived.
        out["foreign_isp_share"] = self._sums["_foreign_isp"] / n
        out["mobile_os_share"] = self._sums["_mobile_os"] / n

        # Temporal.
        out["hours_span"] = ((self._max_ts - self._min_ts) / 3600.0).reindex(out.index).fillna(0.0).clip(lower=0.0)
        out["night_ops_share"] = self._sums["_is_night"] / n
        out["weekend_share"] = self._sums["_is_weekend"] / n

        out = out[FEATURE_COLUMNS].reset_index()
        for c in FEATURE_COLUMNS:
            if c != "event_count":
                out[c] = out[c].astype(np.float32)
        return out


def _iter_batches(paths: Iterable[Path], batch_rows: int = 200_000):
    for p in paths:
        pf = pq.ParquetFile(str(p))
        cols = [c for c in _READ_COLS if c in pf.schema_arrow.names]
        n = pf.metadata.num_rows
        written = 0
        for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
            df = batch.to_pandas()
            for c in _READ_COLS:
                if c not in df.columns:
                    df[c] = np.nan
            yield p.name, df, n, written + len(df)
            written += len(df)


def _resolve_aggregate_paths(
    input_dir: Path | None,
    sources: list[str] | None,
    events_glob: str | Iterable[str] | None,
) -> list[Path]:
    if events_glob:
        patterns = [events_glob] if isinstance(events_glob, str) else list(events_glob)
        paths: list[Path] = []
        for pat in patterns:
            paths.extend(Path(p) for p in sorted(glob.glob(pat)))
        return paths
    if input_dir is None:
        raise ValueError("Either events_glob or input_dir must be provided.")
    if sources is None:
        sources = [
            "pretrain_part_1.parquet", "pretrain_part_2.parquet", "pretrain_part_3.parquet",
            "train_part_1.parquet", "train_part_2.parquet", "train_part_3.parquet",
            "pretest.parquet",
        ]
    return [input_dir / s for s in sources]


def build_customer_features(
    input_dir: Path | None = None,
    output_path: Path = Path("customer_features.parquet"),
    sources: list[str] | None = None,
    *,
    events_glob: str | Iterable[str] | None = None,
    incremental: bool = False,
    state_path: Path | None = None,
    batch_rows: int = 200_000,
    progress: bool = True,
) -> pd.DataFrame:
    """Build per-customer aggregates from event parquets.

    Parameters
    ----------
    input_dir, sources :
        Legacy mode — directory + filename list inside it.
    events_glob :
        Production mode — one or more glob patterns like
        ``"/var/fraud/events/dt=2026-05-*/part-*.parquet"``. Sorted lexically
        so file order is deterministic.
    incremental, state_path :
        If ``incremental=True``, prior raw sums are loaded from ``state_path``
        (created on a previous run via ``CustomerAggregator.save_state``).
        ``paths`` should then contain ONLY new partitions; passing already-
        processed files would double-count them.
    """
    paths = _resolve_aggregate_paths(input_dir, sources, events_glob)
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    if incremental:
        if state_path is None:
            raise ValueError("incremental=True requires state_path.")
        agg = CustomerAggregator.load_state(state_path)
        if progress and agg._sums is not None:
            print(f"[aggregate] resumed state from {state_path} "
                  f"({len(agg._sums):,} customers, "
                  f"{len(agg._processed_files):,} files already folded in)")
    else:
        agg = CustomerAggregator()

    # In incremental mode, skip files whose (path, size, mtime_ns) already
    # appears in the persisted set. In full mode, process everything.
    if incremental:
        new_paths = [p for p in paths if agg.should_process(p)]
        skipped = len(paths) - len(new_paths)
        if progress and skipped:
            print(f"[aggregate] skipping {skipped} file(s) already in state")
        paths = new_paths

    if not paths:
        if progress:
            print("[aggregate] no new input parquets — nothing to process")
    cur_file: str | None = None
    cur_path: Path | None = None
    file_by_name = {p.name: p for p in paths}
    for fname, df, total, written in _iter_batches(paths, batch_rows):
        if fname != cur_file:
            cur_file = fname
            cur_path = file_by_name.get(fname)
            if progress:
                print(f"[aggregate] {fname}: {total:,} rows")
        agg.process(df)
        if progress:
            print(f"  ... {written:,}/{total:,}", end="\r")
        # When we've consumed the last batch of this file, mark it processed.
        if cur_path is not None and written >= total:
            agg.mark_processed(cur_path)
            cur_path = None  # avoid double-marking on next batch boundary
    if progress and cur_file is not None:
        print()

    if incremental and state_path is not None and agg._sums is not None:
        agg.save_state(state_path)
        if progress:
            print(f"[aggregate] saved state -> {state_path} "
                  f"({len(agg._processed_files):,} processed files tracked)")

    # If no new files and no prior state, return empty — don't crash on finalize.
    if agg._sums is None:
        if progress:
            print("[aggregate] no data — writing empty customer_features.parquet")
        empty = pd.DataFrame(columns=["customer_id"] + FEATURE_COLUMNS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(str(output_path), compression="snappy", index=False)
        return empty

    feats = agg.finalize()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(str(output_path), compression="snappy", index=False)
    if progress:
        print(f"[aggregate] wrote {len(feats):,} customers -> {output_path}")
    return feats
