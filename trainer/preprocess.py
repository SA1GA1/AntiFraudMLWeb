"""Preprocessor: словари + z-score, без sklearn-зависимостей. Picklable."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .aggregate import FEATURE_COLUMNS as AGG_FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Схема task.md (web-fraud, 71 признак). Numeric / Categorical для модели.
# ---------------------------------------------------------------------------

NUMERIC_COLS: List[str] = [
    # Source numeric + log-transformed
    "operaton_amt",
    # Бинарные task.md (0/1)
    "is_developer_tools", "is_headless_browser", "is_incognito",
    "is_vpn_detected", "is_proxy_detected", "is_tor_detected",
    "is_new_device", "is_new_browser", "biometric_login",
    # Network / fingerprints
    "network_rtt_avg_ms", "screen_color_depth", "installed_fonts_count",
    "timezone_offset",
    # Mouse
    "mouse_velocity_avg", "mouse_acceleration_avg",
    "mouse_jitter_score", "mouse_linearity_score",
    # Click / scroll
    "click_duration_avg_ms", "right_click_count", "scroll_velocity_avg",
    # Keyboard
    "keyboard_typing_speed_median_ms", "keyboard_typing_speed_std_dev",
    "keyboard_typing_rhythm_cv",
    # Clipboard / interactions
    "backspace_ratio", "clipboard_paste_ratio",
    "copy_events_count", "paste_events_count",
    "tab_switch_count", "focus_blur_count",
    "form_fill_duration_sec", "idle_time_before_submit_sec",
    "error_correction_ratio", "hover_time_avg_ms",
    "double_click_count", "drag_drop_events",
    "resize_events_count", "zoom_level",
    # Session shape
    "session_duration_sec", "pages_visited_count",
    # Login / trust
    "failed_login_attempts", "time_since_last_login_sec",
    "device_trust_score",
    # Network identifiers
    "asn",
]

CATEGORICAL_COLS: List[str] = [
    # Source categorical (task.md)
    "currency_iso_cd", "mcc_code", "pos_cd",
    # Browser identity
    "browser_name", "browser_version", "os_type", "os_version",
    "screen_resolution", "system_language",
    "browser_language", "accept_language",
    # Transaction / merchant
    "merchant_name", "transaction_type",
    # Network
    "connection_type", "isp_name",
    # Fingerprints (high-card but bucketed via vocab cap)
    "webgl_vendor",
    # Login
    "login_method",
    # Temporal
    "hour_of_day", "day_of_week",
]

MAX_VOCAB: int = 1024
_UNK_TOKEN: str = "__UNK__"
_NAN_TOKEN: str = "__NAN__"


def _temporal_categoricals(dttm: pd.Series) -> dict[str, pd.Series]:
    dt = pd.to_datetime(dttm, errors="coerce", utc=False)
    hour = dt.dt.hour.fillna(12).astype(np.int16).astype(str)
    dow = dt.dt.dayofweek.fillna(0).astype(np.int16).astype(str)
    return {"hour_of_day": hour, "day_of_week": dow}


def _prepare_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # log1p amount.
    out["operaton_amt"] = np.log1p(
        pd.to_numeric(out["operaton_amt"], errors="coerce").fillna(0).clip(lower=0)
    )

    # Бинарные флаги.
    for c in ("is_developer_tools", "is_headless_browser", "is_incognito",
              "is_vpn_detected", "is_proxy_detected", "is_tor_detected",
              "is_new_device", "is_new_browser"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.float32).clip(0, 1)
        else:
            out[c] = np.float32(0.0)

    # Производный флаг biometric_login = (login_method == 'biometric').
    if "login_method" in out.columns:
        out["biometric_login"] = (out["login_method"].astype(str).fillna("") == "biometric").astype(np.float32)
    else:
        out["biometric_login"] = np.float32(0.0)

    # Числовые task.md.
    numeric_to_float = [
        "network_rtt_avg_ms", "screen_color_depth", "installed_fonts_count",
        "timezone_offset", "asn",
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
        "failed_login_attempts", "time_since_last_login_sec",
        "device_trust_score",
    ]
    for c in numeric_to_float:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)
        else:
            out[c] = np.float32(0.0)

    # Временные категориальные.
    has_h = "hour_of_day" in out.columns and out["hour_of_day"].notna().any()
    has_d = "day_of_week" in out.columns and out["day_of_week"].notna().any()
    if has_h:
        out["hour_of_day"] = (
            pd.to_numeric(out["hour_of_day"], errors="coerce")
            .fillna(12).astype(np.int16).astype(str)
        )
    if has_d:
        out["day_of_week"] = (
            pd.to_numeric(out["day_of_week"], errors="coerce")
            .fillna(0).astype(np.int16).astype(str)
        )
    if (not has_h or not has_d) and "event_dttm" in out.columns:
        derived = _temporal_categoricals(out["event_dttm"])
        if not has_h:
            out["hour_of_day"] = derived["hour_of_day"]
        if not has_d:
            out["day_of_week"] = derived["day_of_week"]
    if "hour_of_day" not in out.columns:
        out["hour_of_day"] = "12"
    if "day_of_week" not in out.columns:
        out["day_of_week"] = "0"

    # Все категориальные — в строки.
    for c in CATEGORICAL_COLS:
        if c in out.columns:
            out[c] = out[c].astype("string").fillna(_NAN_TOKEN)
        else:
            out[c] = _NAN_TOKEN
    return out


@dataclass
class Preprocessor:
    num_mean: Dict[str, float] = field(default_factory=dict)
    num_std: Dict[str, float] = field(default_factory=dict)
    vocab: Dict[str, Dict[str, int]] = field(default_factory=dict)
    agg_mean: Dict[str, float] = field(default_factory=dict)
    agg_std: Dict[str, float] = field(default_factory=dict)
    n_numeric: int = 0
    n_categorical: int = 0
    n_aggregate: int = 0
    cat_vocab_sizes: List[int] = field(default_factory=list)

    def fit(self, events_df: pd.DataFrame, agg_df: pd.DataFrame | None = None) -> "Preprocessor":
        ev = _prepare_event_columns(events_df)

        for col in NUMERIC_COLS:
            v = pd.to_numeric(ev[col], errors="coerce")
            self.num_mean[col] = float(v.mean())
            std = float(v.std(ddof=0))
            self.num_std[col] = std if std > 1e-6 else 1.0

        for col in CATEGORICAL_COLS:
            counts = ev[col].astype(str).value_counts()
            top = counts.iloc[: MAX_VOCAB - 1].index.tolist()
            self.vocab[col] = {_UNK_TOKEN: 0, **{v: i + 1 for i, v in enumerate(top)}}

        if agg_df is not None:
            for col in AGG_FEATURE_COLUMNS:
                v = pd.to_numeric(agg_df[col], errors="coerce")
                self.agg_mean[col] = float(v.mean())
                std = float(v.std(ddof=0))
                self.agg_std[col] = std if std > 1e-6 else 1.0

        self.n_numeric = len(NUMERIC_COLS)
        self.n_categorical = len(CATEGORICAL_COLS)
        self.n_aggregate = len(AGG_FEATURE_COLUMNS)
        self.cat_vocab_sizes = [len(self.vocab[c]) for c in CATEGORICAL_COLS]
        return self

    def transform_events(self, events_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ev = _prepare_event_columns(events_df)
        n = len(ev)

        num = np.empty((n, len(NUMERIC_COLS)), dtype=np.float32)
        for i, col in enumerate(NUMERIC_COLS):
            v = pd.to_numeric(ev[col], errors="coerce").astype(np.float64).to_numpy()
            v = (v - self.num_mean[col]) / self.num_std[col]
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            num[:, i] = v.astype(np.float32)

        cat = np.empty((n, len(CATEGORICAL_COLS)), dtype=np.int64)
        for i, col in enumerate(CATEGORICAL_COLS):
            mapping = self.vocab[col]
            vals = ev[col].astype(str).to_numpy()
            cat[:, i] = np.array([mapping.get(v, 0) for v in vals], dtype=np.int64)
        return num, cat

    def transform_aggregates(self, customer_ids: np.ndarray, agg_df: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
        n = len(customer_ids)
        agg_arr = np.zeros((n, len(AGG_FEATURE_COLUMNS)), dtype=np.float32)
        has_hist = np.zeros((n, 1), dtype=np.float32)
        if agg_df is None or not self.agg_mean:
            return agg_arr, has_hist
        agg_indexed = agg_df.set_index("customer_id")
        joined = agg_indexed.reindex(customer_ids)
        mask_present = joined[AGG_FEATURE_COLUMNS[0]].notna().to_numpy()
        has_hist[:, 0] = mask_present.astype(np.float32)
        for j, col in enumerate(AGG_FEATURE_COLUMNS):
            v = pd.to_numeric(joined[col], errors="coerce").astype(np.float64).to_numpy()
            v = (v - self.agg_mean[col]) / self.agg_std[col]
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            agg_arr[:, j] = v.astype(np.float32)
        agg_arr[~mask_present] = 0.0
        return agg_arr, has_hist
