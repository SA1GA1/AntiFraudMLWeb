"""Потоковая агрегация клиентских признаков из data_augmented/."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Бинарные task.md колонки → доли (mean)
_BINARY_COLS = [
    "is_rooted_jailbroken", "is_emulator", "is_debugger_attached",
    "developer_tools_enabled", "is_vpn_detected", "is_proxy_detected",
    "biometric_entry_used",
]
# Числовые task.md колонки → mean
_MEAN_COLS = [
    "network_rtt_avg_ms", "accuracy_meters", "geo_speed_km_h",
    "touch_typing_rhythm_cv", "touch_typing_rhythm_median_ms",
    "touch_typing_rhythm_std_dev", "tap_velocity_avg",
    "touch_jitter_score", "swipe_angle_deviation",
    "clipboard_paste_ratio", "backspace_ratio", "form_fill_duration_sec",
    "app_background_events", "screen_orientation_changes",
    "accelerometer_variance_x", "accelerometer_variance_y", "gyroscope_variance",
    "battery_level", "storage_free_percent",
]

FEATURE_COLUMNS = [
    # Amount
    "event_count", "amt_mean", "amt_std", "amt_max", "amt_log_mean",
    # Attestation / security shares
    "is_rooted_share", "is_emulator_share", "is_debugger_share",
    "developer_tools_share", "biometric_share",
    "attestation_failed_share", "integrity_fail_share", "sideload_share",
    # Network
    "vpn_share", "proxy_share", "mean_rtt", "foreign_sim_share",
    # Geo
    "mean_geo_speed", "mean_accuracy",
    # Biometrics (averages)
    "mean_typing_cv", "mean_typing_median_ms", "mean_typing_std",
    "mean_tap_velocity", "mean_jitter", "mean_swipe_angle",
    "mean_clipboard_paste", "mean_backspace", "mean_form_fill",
    "mean_app_background", "mean_orientation",
    "mean_accel_x", "mean_accel_y", "mean_gyro",
    # Device state
    "mean_battery", "mean_storage_free", "charging_24_7_share",
    # Temporal
    "hours_span", "night_ops_share", "weekend_share",
]

_READ_COLS = [
    "customer_id", "event_dttm", "operaton_amt",
    "attestation_status", "integrity_token", "app_install_source",
    "battery_charging_state", "sim_country_code",
] + _BINARY_COLS + _MEAN_COLS


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

    # Derived shares
    if "attestation_status" in df.columns:
        df["_attestation_failed"] = (df["attestation_status"].fillna("passed") != "passed").astype(np.float32)
    else:
        df["_attestation_failed"] = np.float32(0.0)
    if "integrity_token" in df.columns:
        df["_integrity_fail"] = df["integrity_token"].fillna("pass").isin(["fail", "missing"]).astype(np.float32)
    else:
        df["_integrity_fail"] = np.float32(0.0)
    if "app_install_source" in df.columns:
        df["_sideload"] = (df["app_install_source"].fillna("official") == "sideload").astype(np.float32)
    else:
        df["_sideload"] = np.float32(0.0)
    if "sim_country_code" in df.columns:
        df["_foreign_sim"] = (df["sim_country_code"].fillna("RU") != "RU").astype(np.float32)
    else:
        df["_foreign_sim"] = np.float32(0.0)
    if "battery_charging_state" in df.columns:
        df["_charging_24_7"] = (df["battery_charging_state"].fillna("discharging") == "plugged_24_7").astype(np.float32)
    else:
        df["_charging_24_7"] = np.float32(0.0)

    # Temporal
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
    + ["_attestation_failed", "_integrity_fail", "_sideload",
       "_foreign_sim", "_charging_24_7", "_is_night", "_is_weekend"]
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

        # Бинарные task.md → доли
        out["is_rooted_share"] = self._sums["is_rooted_jailbroken"] / n
        out["is_emulator_share"] = self._sums["is_emulator"] / n
        out["is_debugger_share"] = self._sums["is_debugger_attached"] / n
        out["developer_tools_share"] = self._sums["developer_tools_enabled"] / n
        out["biometric_share"] = self._sums["biometric_entry_used"] / n
        out["attestation_failed_share"] = self._sums["_attestation_failed"] / n
        out["integrity_fail_share"] = self._sums["_integrity_fail"] / n
        out["sideload_share"] = self._sums["_sideload"] / n

        # Network
        out["vpn_share"] = self._sums["is_vpn_detected"] / n
        out["proxy_share"] = self._sums["is_proxy_detected"] / n
        out["mean_rtt"] = self._sums["network_rtt_avg_ms"] / n
        out["foreign_sim_share"] = self._sums["_foreign_sim"] / n

        # Geo
        out["mean_geo_speed"] = self._sums["geo_speed_km_h"] / n
        out["mean_accuracy"] = self._sums["accuracy_meters"] / n

        # Biometrics
        out["mean_typing_cv"] = self._sums["touch_typing_rhythm_cv"] / n
        out["mean_typing_median_ms"] = self._sums["touch_typing_rhythm_median_ms"] / n
        out["mean_typing_std"] = self._sums["touch_typing_rhythm_std_dev"] / n
        out["mean_tap_velocity"] = self._sums["tap_velocity_avg"] / n
        out["mean_jitter"] = self._sums["touch_jitter_score"] / n
        out["mean_swipe_angle"] = self._sums["swipe_angle_deviation"] / n
        out["mean_clipboard_paste"] = self._sums["clipboard_paste_ratio"] / n
        out["mean_backspace"] = self._sums["backspace_ratio"] / n
        out["mean_form_fill"] = self._sums["form_fill_duration_sec"] / n
        out["mean_app_background"] = self._sums["app_background_events"] / n
        out["mean_orientation"] = self._sums["screen_orientation_changes"] / n
        out["mean_accel_x"] = self._sums["accelerometer_variance_x"] / n
        out["mean_accel_y"] = self._sums["accelerometer_variance_y"] / n
        out["mean_gyro"] = self._sums["gyroscope_variance"] / n

        # Device state
        out["mean_battery"] = self._sums["battery_level"] / n
        out["mean_storage_free"] = self._sums["storage_free_percent"] / n
        out["charging_24_7_share"] = self._sums["_charging_24_7"] / n

        # Temporal
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
