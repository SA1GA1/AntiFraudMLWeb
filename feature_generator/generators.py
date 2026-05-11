"""Векторизованный детерминированный синтез признаков по схеме task.md.

50 новых колонок. Все случайные значения — SplitMix64-хеш от int-ключа
(customer_id / session_id / event_id) + per-feature salt + global seed.
Повторный запуск с тем же seed → побитово идентичный parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEW_COLUMNS: list[str] = [
    # Identity / device
    "device_id", "installation_id",
    # App / OS
    "app_version", "os_type", "os_version", "device_model",
    # Temporal (derived from event_dttm)
    "hour_of_day", "day_of_week",
    # Transaction
    "merchant_name", "transaction_type",
    # Attestation / security
    "attestation_status", "is_rooted_jailbroken", "is_emulator",
    "is_debugger_attached", "developer_tools_enabled",
    "app_install_source", "integrity_token",
    # Network
    "connection_type", "carrier_name", "carrier_mcc", "carrier_mnc",
    "ip_address_hash", "is_vpn_detected", "is_proxy_detected",
    "network_rtt_avg_ms", "sim_country_code", "sim_carrier_name",
    # Geo
    "latitude", "longitude", "accuracy_meters", "location_provider",
    "timezone_offset_minutes", "geo_speed_km_h",
    # Biometrics
    "touch_typing_rhythm_median_ms", "touch_typing_rhythm_std_dev",
    "touch_typing_rhythm_cv", "tap_velocity_avg", "tap_pressure_avg",
    "touch_jitter_score", "swipe_angle_deviation",
    "clipboard_paste_ratio", "backspace_ratio", "form_fill_duration_sec",
    "app_background_events", "screen_orientation_changes",
    "accelerometer_variance_x", "accelerometer_variance_y",
    "gyroscope_variance", "biometric_entry_used",
    # Device state
    "battery_level", "battery_charging_state", "storage_free_percent",
]

# ---------------------------------------------------------------------------
# Categorical pools (fixed dictionaries)
# ---------------------------------------------------------------------------

_OS_TYPES = np.array(["iOS", "Android"])
_IOS_VERSIONS = np.array(["iOS 15", "iOS 16", "iOS 17", "iOS 18"])
_ANDROID_VERSIONS = np.array(["Android 11", "Android 12", "Android 13", "Android 14", "Android 15"])
_IOS_MODELS = np.array([
    "iPhone SE", "iPhone 12", "iPhone 13", "iPhone 14",
    "iPhone 15", "iPhone 14 Pro", "iPhone 15 Pro",
])
_ANDROID_MODELS = np.array([
    "Samsung Galaxy S22", "Samsung Galaxy S23", "Samsung Galaxy A52",
    "Xiaomi Redmi Note 11", "Xiaomi 13", "Huawei P50", "Google Pixel 7",
    "Realme 9", "OPPO Reno 8", "OnePlus 11",
])
_APP_VERSIONS = np.array([
    "8.10.0", "8.11.0", "8.12.0", "8.13.0", "9.0.0", "9.1.0", "9.2.0",
])
_TRANSACTION_TYPES = np.array(["payment", "p2p", "cash", "transfer"])
_ATTESTATION_LABELS = np.array(["passed", "failed_root", "emulator", "modified_firmware"])
_INSTALL_SOURCES = np.array(["official", "sideload"])
_INTEGRITY_TOKENS = np.array(["pass", "fail", "missing"])
_CONNECTION_TYPES = np.array(["wifi", "cellular", "5g", "4g"])
_LOCATION_PROVIDERS = np.array(["gps", "network", "passive"])
_BATTERY_STATES = np.array(["discharging", "charging", "full", "plugged_24_7"])

# RU carriers with their canonical MNC codes (MCC=250).
_RU_CARRIERS = np.array(["MTS", "Megafon", "Beeline", "Tele2", "Yota", "Tinkoff Mobile"])
_RU_MNC = np.array([1, 2, 99, 20, 11, 62], dtype=np.int32)

_FOREIGN_CARRIERS = np.array(["Vodafone", "T-Mobile DE", "Orange", "China Mobile", "Kcell", "MTS Belarus"])
_FOREIGN_MCC = np.array([262, 262, 208, 460, 401, 257], dtype=np.int32)
_FOREIGN_MNC = np.array([2, 1, 1, 0, 2, 1], dtype=np.int32)
_FOREIGN_COUNTRIES = np.array(["DE", "DE", "FR", "CN", "KZ", "BY"])

_MERCHANTS = np.array([
    "Yandex Eda", "Wildberries", "Ozon", "Pyaterochka", "Magnit",
    "Aeroflot", "Sber Pay", "VK Pay", "Yandex Taxi", "Burger King",
    "McDonalds", "KFC", "Tinkoff", "Apple Pay", "Google Play",
    "Steam", "Netflix", "YouTube Premium", "DNS", "M.Video",
    "Eldorado", "Aliexpress", "Spotify", "PayPal", "Lenta",
    "Auchan", "Perekrestok", "Vkusvill", "Dixy", "Bristol",
    "Krasnoe & Beloe", "Lukoil", "Rosneft", "Gazprom Neft", "Shell",
    "Sportmaster", "Adidas", "Nike", "H&M", "Zara", "Ikea",
    "Leroy Merlin", "Castorama", "Letoile", "Apteka 36.6",
    "Yandex Lavka", "Delivery Club", "Bolt", "Citymobil", "Booking.com",
])


# ---------------------------------------------------------------------------
# Hash + sampling helpers
# ---------------------------------------------------------------------------

def _mix(x: np.ndarray) -> np.ndarray:
    c1 = np.uint64(0x9E3779B97F4A7C15)
    c2 = np.uint64(0xBF58476D1CE4E5B9)
    c3 = np.uint64(0x94D049BB133111EB)
    x = x + c1
    x = (x ^ (x >> np.uint64(30))) * c2
    x = (x ^ (x >> np.uint64(27))) * c3
    return x ^ (x >> np.uint64(31))


def _to_uint64(keys) -> np.ndarray:
    s = pd.Series(keys)
    if s.isna().any():
        s = s.fillna(0)
    return s.to_numpy(dtype=np.int64, copy=False).astype(np.uint64, copy=False)


def _uniform(keys, salt: int, global_seed: int) -> np.ndarray:
    k = _to_uint64(keys)
    salt_u = np.uint64(salt & 0xFFFFFFFFFFFFFFFF)
    seed_u = np.uint64(global_seed & 0xFFFFFFFFFFFFFFFF)
    h = _mix(_mix(k ^ salt_u) ^ seed_u)
    return h.astype(np.float64) / np.float64(2.0 ** 64)


def _int64_id(keys, salt: int, global_seed: int) -> np.ndarray:
    """Stable int64 identifier from a key + salt + seed."""
    k = _to_uint64(keys)
    salt_u = np.uint64(salt & 0xFFFFFFFFFFFFFFFF)
    seed_u = np.uint64(global_seed & 0xFFFFFFFFFFFFFFFF)
    h = _mix(_mix(k ^ salt_u) ^ seed_u)
    return h.astype(np.int64)


def _hex_id(keys, salt: int, global_seed: int, length: int = 12) -> np.ndarray:
    """Stable hex-string id derived from key + salt + seed."""
    k = _to_uint64(keys)
    salt_u = np.uint64(salt & 0xFFFFFFFFFFFFFFFF)
    seed_u = np.uint64(global_seed & 0xFFFFFFFFFFFFFFFF)
    h = _mix(_mix(k ^ salt_u) ^ seed_u)
    fmt = f"%0{length}x"
    masked = (h & np.uint64((1 << (4 * length)) - 1)).tolist()
    return np.array([fmt % v for v in masked], dtype=object)


def _categorical(u: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cum = np.cumsum(probs, axis=1)
    idx = (u[:, None] < cum).argmax(axis=1)
    return labels[idx]


def _uniform_pick(u: np.ndarray, pool: np.ndarray) -> np.ndarray:
    idx = np.floor(u * len(pool)).astype(np.int64)
    idx = np.clip(idx, 0, len(pool) - 1)
    return pool[idx]


# ---------------------------------------------------------------------------
# Risk score
# ---------------------------------------------------------------------------

def _truthy(series: pd.Series) -> np.ndarray:
    if series is None:
        return np.zeros(0, dtype=np.float64)
    if pd.api.types.is_numeric_dtype(series):
        v = series.fillna(0).to_numpy()
        return (v > 0).astype(np.float64)
    s = series.fillna("0").astype(str).str.strip().str.lower()
    return s.isin({"1", "true", "yes", "t", "y"}).to_numpy().astype(np.float64)


def compute_risk_score(df: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    compromised = _truthy(df.get("compromised"))
    dev_tools = _truthy(df.get("developer_tools"))
    rdp = _truthy(df.get("web_rdp_connection"))
    voip = _truthy(df.get("phone_voip_call_state"))
    fallback = np.clip(
        0.10 + 0.30 * compromised + 0.25 * rdp + 0.20 * dev_tools + 0.10 * voip,
        0.05, 0.65,
    )
    risk = fallback.copy()
    if target is not None and len(target) == len(risk):
        is_fraud = target == 1.0
        is_benign = target == 0.0
        risk = np.where(is_fraud, 0.85, risk)
        risk = np.where(is_benign, 0.10, risk)
    return risk.astype(np.float64)


def _lerp(a, b, r):
    return a * (1.0 - r) + b * r


# ---------------------------------------------------------------------------
# Individual generators
# ---------------------------------------------------------------------------

# ---- Identity ----

def _gen_device_id(customer_id, seed):
    return _int64_id(customer_id, salt=0xDE1CE10D, global_seed=seed)


def _gen_installation_id(customer_id, event_id, risk, seed):
    """Stable per customer normally; for ~5% of high-risk customers, drifts
    occasionally (simulating reinstalls / dropper rotation)."""
    base = _int64_id(customer_id, salt=0x12574117, global_seed=seed)
    u_drift = _uniform(event_id, salt=0x12574118, global_seed=seed)
    p_drift = _lerp(0.0, 0.10, risk)
    drifted = _int64_id(np.bitwise_xor(_to_uint64(customer_id),
                                       _to_uint64(event_id // 13)),
                        salt=0x12574119, global_seed=seed)
    return np.where(u_drift < p_drift, drifted, base)


# ---- App / OS ----

def _gen_os_type(customer_id, seed):
    u = _uniform(customer_id, salt=0x05057E00, global_seed=seed)
    return np.where(u < 0.55, "Android", "iOS").astype(object)


def _gen_os_version(customer_id, os_type, seed):
    u = _uniform(customer_id, salt=0x05057E01, global_seed=seed)
    ios_pick = _uniform_pick(u, _IOS_VERSIONS)
    android_pick = _uniform_pick(u, _ANDROID_VERSIONS)
    return np.where(os_type == "iOS", ios_pick, android_pick).astype(object)


def _gen_device_model(customer_id, os_type, seed):
    u = _uniform(customer_id, salt=0x0DE7E110, global_seed=seed)
    ios_pick = _uniform_pick(u, _IOS_MODELS)
    android_pick = _uniform_pick(u, _ANDROID_MODELS)
    return np.where(os_type == "iOS", ios_pick, android_pick).astype(object)


def _gen_app_version(customer_id, seed):
    u = _uniform(customer_id, salt=0x4990E751, global_seed=seed)
    return _uniform_pick(u, _APP_VERSIONS).astype(object)


# ---- Transaction ----

def _gen_merchant_name(event_id, seed):
    u = _uniform(event_id, salt=0x3E6CA42E, global_seed=seed)
    return _uniform_pick(u, _MERCHANTS).astype(object)


def _gen_transaction_type(event_id, risk, seed):
    u = _uniform(event_id, salt=0x77A45AC7, global_seed=seed)
    base = np.array([0.55, 0.20, 0.10, 0.15])   # payment, p2p, cash, transfer
    fraud = np.array([0.15, 0.40, 0.10, 0.35])
    probs = base[None, :] * (1 - risk[:, None]) + fraud[None, :] * risk[:, None]
    return _categorical(u, probs, _TRANSACTION_TYPES)


# ---- Attestation / security ----

def _gen_attestation_status(customer_id, risk, compromised, seed):
    u = _uniform(customer_id, salt=0xA11E57A1, global_seed=seed)
    p_pass = _lerp(0.95, 0.20, risk) - 0.25 * compromised
    p_pass = np.clip(p_pass, 0.05, 0.99)
    rest = 1.0 - p_pass
    p_fail = rest * (0.50 - 0.15 * compromised)
    p_emu = rest * (0.20 + 0.10 * compromised)
    p_mod = rest * (0.30 + 0.05 * compromised)
    probs = np.column_stack([p_pass, p_fail, p_emu, p_mod])
    return _categorical(u, probs, _ATTESTATION_LABELS)


def _gen_is_rooted_jailbroken(customer_id, compromised, risk, seed):
    """Mirror source compromised; for events without flag, low base rate
    biased up by risk."""
    u = _uniform(customer_id, salt=0x900754ED, global_seed=seed)
    p = _lerp(0.01, 0.40, risk)
    base = (u < p).astype(np.int8)
    return np.maximum(base, compromised.astype(np.int8))


def _gen_is_emulator(customer_id, attestation, seed):
    """1 if attestation says emulator, else low base rate per device."""
    u = _uniform(customer_id, salt=0xE10500A7, global_seed=seed)
    base = (u < 0.02).astype(np.int8)
    from_attestation = (attestation == "emulator").astype(np.int8)
    return np.maximum(base, from_attestation)


def _gen_is_debugger_attached(event_id, risk, dev_tools, seed):
    u = _uniform(event_id, salt=0xDEB66E12, global_seed=seed)
    p = _lerp(0.005, 0.35, risk) + 0.30 * dev_tools
    return (u < np.clip(p, 0, 1)).astype(np.int8)


def _gen_developer_tools_enabled(dev_tools_source, customer_id, risk, seed):
    """Mirror source developer_tools where present; else low base rate."""
    u = _uniform(customer_id, salt=0xDE7E10F5, global_seed=seed)
    p = _lerp(0.02, 0.30, risk)
    base = (u < p).astype(np.int8)
    return np.maximum(base, dev_tools_source.astype(np.int8))


def _gen_app_install_source(customer_id, risk, seed):
    u = _uniform(customer_id, salt=0x1A570055, global_seed=seed)
    p_sideload = _lerp(0.03, 0.45, risk)
    return np.where(u < p_sideload, "sideload", "official").astype(object)


def _gen_integrity_token(event_id, is_rooted, is_emulator, risk, seed):
    u = _uniform(event_id, salt=0x171E6817, global_seed=seed)
    # Если устройство явно скомпрометировано — pass редко.
    bad = (is_rooted + is_emulator) > 0
    p_pass = np.where(bad, 0.10, _lerp(0.95, 0.55, risk))
    p_fail = np.where(bad, 0.60, _lerp(0.03, 0.30, risk))
    p_missing = 1.0 - p_pass - p_fail
    probs = np.column_stack([p_pass, p_fail, np.clip(p_missing, 0.001, None)])
    return _categorical(u, probs, _INTEGRITY_TOKENS)


# ---- Network ----

def _gen_connection_type(session_id, seed):
    u = _uniform(session_id, salt=0xC0117EC7, global_seed=seed)
    probs = np.array([[0.45, 0.10, 0.30, 0.15]])  # wifi, cellular, 5g, 4g
    return _categorical(u, np.repeat(probs, len(session_id), axis=0), _CONNECTION_TYPES)


def _gen_carrier(customer_id, risk, seed):
    """Returns (carrier_name, mcc, mnc, sim_country_code, sim_carrier_name)."""
    u_country = _uniform(customer_id, salt=0xCA771E60, global_seed=seed)
    u_pick = _uniform(customer_id, salt=0xCA771E61, global_seed=seed)
    is_foreign = u_country < _lerp(0.03, 0.45, risk)

    ru_idx = np.floor(u_pick * len(_RU_CARRIERS)).astype(np.int64).clip(0, len(_RU_CARRIERS) - 1)
    foreign_idx = np.floor(u_pick * len(_FOREIGN_CARRIERS)).astype(np.int64).clip(0, len(_FOREIGN_CARRIERS) - 1)

    carrier_name = np.where(is_foreign, _FOREIGN_CARRIERS[foreign_idx], _RU_CARRIERS[ru_idx]).astype(object)
    mcc = np.where(is_foreign, _FOREIGN_MCC[foreign_idx], np.int32(250)).astype(np.int32)
    mnc = np.where(is_foreign, _FOREIGN_MNC[foreign_idx], _RU_MNC[ru_idx]).astype(np.int32)
    sim_country = np.where(is_foreign, _FOREIGN_COUNTRIES[foreign_idx], "RU").astype(object)
    sim_carrier = carrier_name  # часто совпадает в реальности
    return carrier_name, mcc, mnc, sim_country, sim_carrier


def _gen_ip_address_hash(session_id, seed):
    """int64 to save disk vs hex string; semantically still a stable hash."""
    return _int64_id(session_id, salt=0x19A4D255, global_seed=seed)


def _gen_is_vpn_detected(session_id, risk, seed):
    u = _uniform(session_id, salt=0xF6E700D1, global_seed=seed)
    p = _lerp(0.02, 0.40, risk)
    return (u < p).astype(np.int8)


def _gen_is_proxy_detected(session_id, risk, seed):
    u = _uniform(session_id, salt=0xF6E700D2, global_seed=seed)
    p = _lerp(0.01, 0.25, risk)
    return (u < p).astype(np.int8)


def _gen_network_rtt_avg_ms(session_id, risk, rdp, seed):
    u = _uniform(session_id, salt=0x47B12A11, global_seed=seed)
    mean = _lerp(25.0, 220.0, risk) + 60.0 * rdp
    val = -mean * np.log(np.clip(1.0 - u, 1e-9, 1.0))
    return val.astype(np.float32)


# ---- Geo ----

def _gen_geo(customer_id, event_id, risk, seed):
    """Возвращает latitude, longitude, accuracy_meters, location_provider, geo_speed_km_h."""
    # Home location per customer.
    u_lat = _uniform(customer_id, salt=0x6E01A100, global_seed=seed)
    u_lon = _uniform(customer_id, salt=0x6E01A101, global_seed=seed)
    home_lat = 50.0 + u_lat * 20.0           # 50-70 (RU range)
    home_lon = 30.0 + u_lon * 60.0           # 30-90

    # Per-event drift around home.
    u_dlat = _uniform(event_id, salt=0x6E01A110, global_seed=seed)
    u_dlon = _uniform(event_id, salt=0x6E01A111, global_seed=seed)
    drift_scale = _lerp(0.02, 0.30, risk)    # фрод-сессии могут "прыгать" дальше
    lat = (home_lat + (u_dlat - 0.5) * drift_scale * 5.0).astype(np.float32)
    lon = (home_lon + (u_dlon - 0.5) * drift_scale * 5.0).astype(np.float32)

    # Accuracy: GPS чаще точный (3-20m), network coarser (30-200m), passive (50-500m).
    u_prov = _uniform(event_id, salt=0x6E01A120, global_seed=seed)
    p_gps = _lerp(0.75, 0.35, risk)
    p_net = _lerp(0.20, 0.45, risk)
    probs = np.column_stack([p_gps, p_net, np.clip(1.0 - p_gps - p_net, 0.01, None)])
    provider = _categorical(u_prov, probs, _LOCATION_PROVIDERS)

    u_acc = _uniform(event_id, salt=0x6E01A130, global_seed=seed)
    base_acc = np.where(provider == "gps", 5.0 + u_acc * 25.0,
                np.where(provider == "network", 30.0 + u_acc * 200.0,
                         50.0 + u_acc * 450.0))
    accuracy = (base_acc * _lerp(1.0, 2.5, risk)).astype(np.float32)

    # Speed: норма 0-10, фрод может прыгать → 50-300.
    u_speed = _uniform(event_id, salt=0x6E01A140, global_seed=seed)
    mean_speed = _lerp(3.0, 80.0, risk)
    geo_speed = (-mean_speed * np.log(np.clip(1.0 - u_speed, 1e-9, 1.0))).astype(np.float32)
    return lat, lon, accuracy, provider, geo_speed


def _gen_timezone_offset_minutes(customer_id, source_timezone, seed):
    """Если есть source `timezone` — используем; иначе генерируем."""
    if source_timezone is not None:
        v = pd.to_numeric(source_timezone, errors="coerce")
        if v.notna().any():
            return v.fillna(180).astype(np.int32).to_numpy()
    u = _uniform(customer_id, salt=0x721E20FF, global_seed=seed)
    return ((u * 24 - 12).astype(np.int32) * 60)


# ---- Biometrics ----

def _gen_typing_metrics(session_id, risk, seed):
    """Returns median_ms, std_dev, cv, tap_velocity, tap_pressure, jitter, swipe_angle."""
    u_med = _uniform(session_id, salt=0x77491400, global_seed=seed)
    u_std = _uniform(session_id, salt=0x77491401, global_seed=seed)
    u_vel = _uniform(session_id, salt=0x77491402, global_seed=seed)
    u_press = _uniform(session_id, salt=0x77491403, global_seed=seed)
    u_jit = _uniform(session_id, salt=0x77491404, global_seed=seed)
    u_sw = _uniform(session_id, salt=0x77491405, global_seed=seed)
    u_bot = _uniform(session_id, salt=0x77491406, global_seed=seed)

    is_bot = u_bot < risk * 0.7

    median_human = 120.0 + u_med * 200.0    # 120-320 ms
    median_bot = 30.0 + u_med * 30.0        # 30-60 ms (очень быстро)
    median_ms = np.where(is_bot, median_bot, median_human).astype(np.float32)

    std_human = 30.0 + u_std * 70.0
    std_bot = 1.0 + u_std * 4.0
    std_dev = np.where(is_bot, std_bot, std_human).astype(np.float32)

    cv = (std_dev / np.maximum(median_ms, 1.0)).astype(np.float32)

    tap_velocity_human = 200.0 + u_vel * 600.0   # px/s
    tap_velocity_bot = 50.0 + u_vel * 100.0
    tap_velocity = np.where(is_bot, tap_velocity_bot, tap_velocity_human).astype(np.float32)

    pressure_human = 0.3 + u_press * 0.5    # normalised
    pressure_bot = 0.05 + u_press * 0.1     # bots/emulators report low/none
    tap_pressure = np.where(is_bot, pressure_bot, pressure_human).astype(np.float32)

    jitter_human = 0.4 + u_jit * 0.5
    jitter_bot = 0.0 + u_jit * 0.1
    touch_jitter = np.where(is_bot, jitter_bot, jitter_human).astype(np.float32)

    swipe_human = 5.0 + u_sw * 25.0   # degrees
    swipe_bot = 0.5 + u_sw * 2.0
    swipe_angle = np.where(is_bot, swipe_bot, swipe_human).astype(np.float32)

    return median_ms, std_dev, cv, tap_velocity, tap_pressure, touch_jitter, swipe_angle


def _gen_clipboard_paste_ratio(session_id, risk, seed):
    u = _uniform(session_id, salt=0xC1B0A57E, global_seed=seed)
    centre = _lerp(0.10, 0.70, risk)
    return np.clip(centre + (u - 0.5) * 0.40, 0.0, 1.0).astype(np.float32)


def _gen_backspace_ratio(session_id, risk, seed):
    u = _uniform(session_id, salt=0xBAC759A5, global_seed=seed)
    # Человек чаще ошибается → выше; бот идеален → 0.
    centre = _lerp(0.18, 0.04, risk)
    return np.clip(centre + (u - 0.5) * 0.12, 0.0, 1.0).astype(np.float32)


def _gen_form_fill_duration_sec(session_id, risk, seed):
    u = _uniform(session_id, salt=0xF02DF111, global_seed=seed)
    mean = _lerp(80.0, 8.0, risk)
    val = -mean * np.log(np.clip(1.0 - u, 1e-9, 1.0))
    return val.astype(np.float32)


def _gen_app_background_events(session_id, risk, rdp, seed):
    u = _uniform(session_id, salt=0xB6CE0001, global_seed=seed)
    mean = _lerp(0.4, 4.5, risk) + 2.0 * rdp
    return np.floor(-mean * np.log(1.0 - u)).astype(np.int32)


def _gen_screen_orientation_changes(session_id, risk, rdp, seed):
    u = _uniform(session_id, salt=0x05CCA1E0, global_seed=seed)
    mean = _lerp(0.3, 6.0, risk) + 3.0 * rdp
    return np.floor(-mean * np.log(np.clip(1.0 - u, 1e-9, 1.0))).astype(np.int32)


def _gen_sensor_variances(session_id, risk, seed):
    """Returns accel_x, accel_y, gyro variances. Эмуляторы/RDP → 0."""
    u_ax = _uniform(session_id, salt=0x5E450AC1, global_seed=seed)
    u_ay = _uniform(session_id, salt=0x5E450AC2, global_seed=seed)
    u_gy = _uniform(session_id, salt=0x5E450AC3, global_seed=seed)
    u_dead = _uniform(session_id, salt=0x5E450AC0, global_seed=seed)
    is_dead = u_dead < risk * 0.5
    ax = np.where(is_dead, 0.0 + u_ax * 0.01, 0.5 + u_ax * 2.0).astype(np.float32)
    ay = np.where(is_dead, 0.0 + u_ay * 0.01, 0.5 + u_ay * 2.0).astype(np.float32)
    gy = np.where(is_dead, 0.0 + u_gy * 0.001, 0.1 + u_gy * 0.5).astype(np.float32)
    return ax, ay, gy


def _gen_biometric_entry_used(customer_id, event_id, risk, seed):
    cap_u = _uniform(customer_id, salt=0xB10CA9AB, global_seed=seed)
    capable = cap_u < 0.60
    use_u = _uniform(event_id, salt=0xB10CB101, global_seed=seed)
    p_use = _lerp(0.85, 0.10, risk)
    return (capable & (use_u < p_use)).astype(np.int8)


# ---- Device state ----

def _gen_battery_level(event_id, source_battery, seed):
    """Парсим source battery (string), нормализуем в 0-100, иначе генерируем."""
    if source_battery is not None and len(source_battery) > 0:
        v = pd.to_numeric(source_battery, errors="coerce")
        if v.notna().any():
            arr = v.to_numpy().astype(np.float64)
            arr = np.where(arr <= 1.5, arr * 100.0, arr)
            arr = np.clip(arr, 0.0, 100.0)
            mask_na = np.isnan(arr)
            if mask_na.any():
                u = _uniform(event_id, salt=0xBA77E251, global_seed=seed)
                arr = np.where(mask_na, 20.0 + u * 70.0, arr)
            return arr.astype(np.float32)
    u = _uniform(event_id, salt=0xBA77E251, global_seed=seed)
    return (20.0 + u * 70.0).astype(np.float32)


def _gen_battery_charging_state(event_id, battery_level, risk, seed):
    u = _uniform(event_id, salt=0xC8A36100, global_seed=seed)
    full_bias = np.clip((battery_level - 90.0) / 10.0, 0.0, 1.0)
    base = np.array([0.50, 0.25, 0.20, 0.05])
    fraud = np.array([0.20, 0.20, 0.20, 0.40])
    probs = base[None, :] * (1.0 - risk[:, None]) + fraud[None, :] * risk[:, None]
    shift = full_bias * 0.30
    probs[:, 0] = probs[:, 0] - shift
    probs[:, 2] = probs[:, 2] + shift
    return _categorical(u, probs, _BATTERY_STATES)


def _gen_temporal(event_dttm) -> tuple[np.ndarray, np.ndarray]:
    """Hour-of-day (0..23) and day-of-week (0..6) from event_dttm."""
    dt = pd.to_datetime(event_dttm, errors="coerce", utc=False)
    hour = dt.dt.hour.fillna(12).astype(np.int8).to_numpy()
    dow = dt.dt.dayofweek.fillna(0).astype(np.int8).to_numpy()
    return hour, dow


def _gen_storage_free_percent(customer_id, event_id, risk, seed):
    base_u = _uniform(customer_id, salt=0x5707A6E0, global_seed=seed)
    drift_u = _uniform(event_id, salt=0x5707A6E1, global_seed=seed)
    base = _lerp(75.0, 18.0, risk) + (base_u - 0.5) * 25.0
    val = base + (drift_u - 0.5) * 5.0
    return np.clip(val, 0.5, 99.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_features(df: pd.DataFrame, target: np.ndarray | None, seed: int = 42) -> pd.DataFrame:
    n = len(df)
    if target is None:
        target = np.full(n, np.nan, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert len(target) == n

    customer_id = df["customer_id"].to_numpy()
    event_id = df["event_id"].to_numpy()
    if "session_id" in df.columns:
        s = df["session_id"]
        session_id = s.where(s.notna(), df["event_id"]).to_numpy()
    else:
        session_id = event_id

    compromised = _truthy(df.get("compromised"))
    rdp = _truthy(df.get("web_rdp_connection"))
    dev_tools_src = _truthy(df.get("developer_tools"))
    risk = compute_risk_score(df, target)

    out = pd.DataFrame(index=df.index)

    # Identity
    out["device_id"] = _gen_device_id(customer_id, seed)
    out["installation_id"] = _gen_installation_id(customer_id, event_id, risk, seed)

    # App / OS
    out["app_version"] = _gen_app_version(customer_id, seed)
    out["os_type"] = _gen_os_type(customer_id, seed)
    out["os_version"] = _gen_os_version(customer_id, out["os_type"].to_numpy(), seed)
    out["device_model"] = _gen_device_model(customer_id, out["os_type"].to_numpy(), seed)

    # Transaction
    out["merchant_name"] = _gen_merchant_name(event_id, seed)
    out["transaction_type"] = _gen_transaction_type(event_id, risk, seed)

    # Attestation / security
    out["attestation_status"] = _gen_attestation_status(customer_id, risk, compromised, seed)
    out["is_rooted_jailbroken"] = _gen_is_rooted_jailbroken(customer_id, compromised, risk, seed)
    out["is_emulator"] = _gen_is_emulator(customer_id, out["attestation_status"].to_numpy(), seed)
    out["is_debugger_attached"] = _gen_is_debugger_attached(event_id, risk, dev_tools_src, seed)
    out["developer_tools_enabled"] = _gen_developer_tools_enabled(dev_tools_src, customer_id, risk, seed)
    out["app_install_source"] = _gen_app_install_source(customer_id, risk, seed)
    out["integrity_token"] = _gen_integrity_token(
        event_id,
        out["is_rooted_jailbroken"].to_numpy(),
        out["is_emulator"].to_numpy(),
        risk, seed,
    )

    # Network
    out["connection_type"] = _gen_connection_type(session_id, seed)
    carrier_name, c_mcc, c_mnc, sim_country, sim_carrier = _gen_carrier(customer_id, risk, seed)
    out["carrier_name"] = carrier_name
    out["carrier_mcc"] = c_mcc
    out["carrier_mnc"] = c_mnc
    out["ip_address_hash"] = _gen_ip_address_hash(session_id, seed)
    out["is_vpn_detected"] = _gen_is_vpn_detected(session_id, risk, seed)
    out["is_proxy_detected"] = _gen_is_proxy_detected(session_id, risk, seed)
    out["network_rtt_avg_ms"] = _gen_network_rtt_avg_ms(session_id, risk, rdp, seed)
    out["sim_country_code"] = sim_country
    out["sim_carrier_name"] = sim_carrier

    # Geo
    lat, lon, acc, prov, gs = _gen_geo(customer_id, event_id, risk, seed)
    out["latitude"] = lat
    out["longitude"] = lon
    out["accuracy_meters"] = acc
    out["location_provider"] = prov
    out["timezone_offset_minutes"] = _gen_timezone_offset_minutes(customer_id, df.get("timezone"), seed)
    out["geo_speed_km_h"] = gs

    # Biometrics
    median_ms, std_dev, cv, tap_v, tap_p, jit, sw = _gen_typing_metrics(session_id, risk, seed)
    out["touch_typing_rhythm_median_ms"] = median_ms
    out["touch_typing_rhythm_std_dev"] = std_dev
    out["touch_typing_rhythm_cv"] = cv
    out["tap_velocity_avg"] = tap_v
    out["tap_pressure_avg"] = tap_p
    out["touch_jitter_score"] = jit
    out["swipe_angle_deviation"] = sw
    out["clipboard_paste_ratio"] = _gen_clipboard_paste_ratio(session_id, risk, seed)
    out["backspace_ratio"] = _gen_backspace_ratio(session_id, risk, seed)
    out["form_fill_duration_sec"] = _gen_form_fill_duration_sec(session_id, risk, seed)
    out["app_background_events"] = _gen_app_background_events(session_id, risk, rdp, seed)
    out["screen_orientation_changes"] = _gen_screen_orientation_changes(session_id, risk, rdp, seed)
    ax, ay, gyv = _gen_sensor_variances(session_id, risk, seed)
    out["accelerometer_variance_x"] = ax
    out["accelerometer_variance_y"] = ay
    out["gyroscope_variance"] = gyv
    out["biometric_entry_used"] = _gen_biometric_entry_used(customer_id, event_id, risk, seed)

    # Device state
    out["battery_level"] = _gen_battery_level(event_id, df.get("battery"), seed)
    out["battery_charging_state"] = _gen_battery_charging_state(event_id, out["battery_level"].to_numpy(), risk, seed)
    out["storage_free_percent"] = _gen_storage_free_percent(customer_id, event_id, risk, seed)

    # Temporal (deterministic from event_dttm)
    if "event_dttm" in df.columns:
        hour, dow = _gen_temporal(df["event_dttm"])
    else:
        hour = np.full(n, 12, dtype=np.int8)
        dow = np.full(n, 0, dtype=np.int8)
    out["hour_of_day"] = hour
    out["day_of_week"] = dow

    return out[NEW_COLUMNS]
