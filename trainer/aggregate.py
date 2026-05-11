"""Потоковая агрегация клиентских признаков из data_augmented/.

Все колонки соответствуют web-fraud схеме task.md (71 признак).
"""

from __future__ import annotations

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
    sec_dt = dt.astype("datetime64[s]", copy=False).to_numpy().view("int64")
    df["_ts"] = np.where(valid, sec_dt, np.nan).astype(np.float64)
    return df


_SUM_COLS_FROM_BATCH = (
    ["operaton_amt", "_amt_sq", "_amt_log"]
    + _BINARY_COLS
    + _MEAN_COLS
    + ["_foreign_isp", "_mobile_os", "_is_night", "_is_weekend"]
)


class CustomerAggregator:
    def __init__(self) -> None:
        self._sums: pd.DataFrame | None = None
        self._max_amt: pd.Series | None = None
        self._min_ts: pd.Series | None = None
        self._max_ts: pd.Series | None = None

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


def build_customer_features(
    input_dir: Path,
    output_path: Path,
    sources: list[str] | None = None,
    batch_rows: int = 200_000,
    progress: bool = True,
) -> pd.DataFrame:
    if sources is None:
        sources = [
            "pretrain_part_1.parquet", "pretrain_part_2.parquet", "pretrain_part_3.parquet",
            "train_part_1.parquet", "train_part_2.parquet", "train_part_3.parquet",
            "pretest.parquet",
        ]
    paths = [input_dir / s for s in sources]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    agg = CustomerAggregator()
    cur_file = None
    for fname, df, total, written in _iter_batches(paths, batch_rows):
        if progress and fname != cur_file:
            cur_file = fname
            print(f"[aggregate] {fname}: {total:,} rows")
        agg.process(df)
        if progress:
            print(f"  ... {written:,}/{total:,}", end="\r")
    if progress:
        print()
    feats = agg.finalize()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(str(output_path), compression="snappy", index=False)
    if progress:
        print(f"[aggregate] wrote {len(feats):,} customers -> {output_path}")
    return feats
