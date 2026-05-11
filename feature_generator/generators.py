"""Detereministic feature synthesis aligned with task.md (71 web-fraud columns).

Все случайные значения — SplitMix64-хеш от int-ключа
(customer_id / session_id / event_id) + per-feature salt + global seed.
Повторный запуск с тем же seed → побитово идентичный parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 71 колонка из task.md (порядок сохраняем).
TASK_COLUMNS: list[str] = [
    "customer_id", "event_id", "session_id",
    "browser_fingerprint", "user_agent", "browser_name", "browser_version",
    "os_type", "os_version",
    "event_dttm", "hour_of_day", "day_of_week",
    "operaton_amt", "currency_iso_cd", "mcc_code",
    "merchant_name", "pos_cd", "transaction_type",
    "is_developer_tools", "is_headless_browser",
    "screen_resolution", "screen_color_depth",
    "timezone_offset", "system_language", "browser_language", "accept_language",
    "is_incognito", "installed_fonts_count",
    "ip_address_hash", "is_vpn_detected", "is_proxy_detected", "is_tor_detected",
    "connection_type", "network_rtt_avg_ms",
    "asn", "isp_name",
    "mouse_velocity_avg", "mouse_acceleration_avg",
    "mouse_jitter_score", "mouse_linearity_score",
    "click_duration_avg_ms", "right_click_count",
    "scroll_velocity_avg",
    "keyboard_typing_speed_median_ms", "keyboard_typing_speed_std_dev",
    "keyboard_typing_rhythm_cv",
    "backspace_ratio", "clipboard_paste_ratio",
    "copy_events_count", "paste_events_count",
    "tab_switch_count", "focus_blur_count",
    "form_fill_duration_sec", "idle_time_before_submit_sec",
    "error_correction_ratio", "hover_time_avg_ms",
    "double_click_count", "drag_drop_events",
    "resize_events_count", "zoom_level",
    "webgl_vendor", "canvas_fingerprint", "audio_fingerprint",
    "session_duration_sec", "pages_visited_count",
    "login_method", "failed_login_attempts", "time_since_last_login_sec",
    "is_new_device", "is_new_browser", "device_trust_score",
]

# Обратная совместимость со старым импортом.
NEW_COLUMNS = TASK_COLUMNS


# ---------------------------------------------------------------------------
# Категориальные пулы
# ---------------------------------------------------------------------------

_BROWSER_NAMES = np.array(["Chrome", "Firefox", "Safari", "Edge", "Opera", "Yandex"])
_BROWSER_VERSIONS = {
    "Chrome": np.array(["118.0", "119.0", "120.0", "121.0", "122.0", "123.0", "124.0"]),
    "Firefox": np.array(["115.0", "118.0", "119.0", "120.0", "121.0", "122.0"]),
    "Safari": np.array(["16.1", "16.3", "16.5", "17.0", "17.2", "17.4"]),
    "Edge": np.array(["118.0", "119.0", "120.0", "121.0", "122.0"]),
    "Opera": np.array(["100.0", "101.0", "102.0", "103.0", "104.0"]),
    "Yandex": np.array(["23.7", "23.9", "23.11", "24.1", "24.3"]),
}
_OS_TYPES = np.array(["Windows", "macOS", "Linux", "Android", "iOS"])
_OS_VERSIONS = {
    "Windows": np.array(["10", "11"]),
    "macOS": np.array(["12", "13", "14"]),
    "Linux": np.array(["Ubuntu 22.04", "Fedora 39", "Arch", "Debian 12"]),
    "Android": np.array(["11", "12", "13", "14"]),
    "iOS": np.array(["15", "16", "17"]),
}
_SCREEN_RES_DESKTOP = np.array([
    "1920x1080", "2560x1440", "1366x768", "1536x864", "1440x900",
    "3840x2160", "1680x1050", "2880x1800",
])
_SCREEN_RES_MOBILE = np.array([
    "1170x2532", "1284x2778", "1080x2400", "1080x1920", "720x1280",
    "1440x3200", "828x1792", "1242x2688",
])
_COLOR_DEPTHS = np.array([24, 30, 32], dtype=np.int16)
_TRANSACTION_TYPES = np.array(["payment", "p2p", "transfer", "cash", "refund"])
_CONNECTION_TYPES = np.array(["wifi", "cellular", "ethernet", "5g", "4g", "unknown"])
_LOGIN_METHODS = np.array(["password", "biometric", "otp", "sso"])
_SYSTEM_LANGS = np.array(["ru-RU", "en-US", "en-GB", "de-DE", "fr-FR", "zh-CN", "kk-KZ", "be-BY"])
_WEBGL_VENDORS = np.array([
    "Intel Inc.", "NVIDIA Corporation", "AMD", "Apple Inc.",
    "Google Inc. (Intel)", "Google Inc. (NVIDIA)", "Qualcomm",
])
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
_ISP_RU = np.array([
    "Rostelecom", "MTS", "Beeline", "MegaFon", "Tele2", "ER-Telecom", "Tinkoff Mobile",
    "TTK", "Sovintel", "Yota",
])
_ISP_FOREIGN = np.array([
    "Vodafone DE", "T-Mobile DE", "Orange FR", "China Mobile", "China Telecom",
    "AT&T", "Comcast", "Kcell", "Beltelecom", "Turk Telekom",
])
_ASN_RU = np.array([8359, 25513, 12389, 8402, 39134, 31133, 41682, 8492, 12714, 21351], dtype=np.int32)
_ASN_FOREIGN = np.array([3209, 3320, 5511, 9808, 4134, 7018, 7922, 49505, 6697, 9121], dtype=np.int32)


# ---------------------------------------------------------------------------
# Хеш-помощники
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


def _hash(keys, salt: int, global_seed: int) -> np.ndarray:
    k = _to_uint64(keys)
    salt_u = np.uint64(salt & 0xFFFFFFFFFFFFFFFF)
    seed_u = np.uint64(global_seed & 0xFFFFFFFFFFFFFFFF)
    return _mix(_mix(k ^ salt_u) ^ seed_u)


def _uniform(keys, salt: int, global_seed: int) -> np.ndarray:
    return _hash(keys, salt, global_seed).astype(np.float64) / np.float64(2.0 ** 64)


def _hex_id(keys, salt: int, global_seed: int, length: int = 16) -> np.ndarray:
    h = _hash(keys, salt, global_seed)
    fmt = f"%0{length}x"
    mask = np.uint64((1 << min(64, 4 * length)) - 1) if length < 16 else np.uint64(0xFFFFFFFFFFFFFFFF)
    masked = (h & mask).tolist()
    return np.array([fmt % v for v in masked], dtype=object)


def _categorical(u: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cum = np.cumsum(probs, axis=1)
    idx = (u[:, None] < cum).argmax(axis=1)
    return labels[idx]


def _pick(u: np.ndarray, pool: np.ndarray) -> np.ndarray:
    idx = np.floor(u * len(pool)).astype(np.int64)
    idx = np.clip(idx, 0, len(pool) - 1)
    return pool[idx]


def _lerp(a, b, r):
    return a * (1.0 - r) + b * r


def _exp_sample(u: np.ndarray, mean) -> np.ndarray:
    return -np.asarray(mean) * np.log(np.clip(1.0 - u, 1e-9, 1.0))


# ---------------------------------------------------------------------------
# Risk score
# ---------------------------------------------------------------------------

def _truthy(series: pd.Series | None) -> np.ndarray:
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
    n = len(df)
    if len(compromised) != n:
        compromised = np.zeros(n)
    if len(dev_tools) != n:
        dev_tools = np.zeros(n)
    if len(rdp) != n:
        rdp = np.zeros(n)
    if len(voip) != n:
        voip = np.zeros(n)
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


# ---------------------------------------------------------------------------
# Browser identity & device
# ---------------------------------------------------------------------------

def _gen_os_type(customer_id, source_os, seed) -> np.ndarray:
    if source_os is not None:
        v = pd.to_numeric(source_os, errors="coerce")
        if v.notna().any():
            mapping = np.array(["Windows", "Android", "iOS", "macOS", "Linux"])
            idx = v.fillna(0).astype(int).clip(0, len(mapping) - 1).to_numpy()
            return mapping[idx].astype(object)
    u = _uniform(customer_id, salt=0x05057E00, global_seed=seed)
    probs = np.array([[0.35, 0.10, 0.05, 0.30, 0.20]])  # Windows/macOS/Linux/Android/iOS
    return _categorical(u, np.repeat(probs, len(customer_id), axis=0), _OS_TYPES)


def _gen_os_version(customer_id, os_type, source_ver, seed) -> np.ndarray:
    if source_ver is not None:
        s = source_ver.astype(str).fillna("")
        has = s.str.len() > 0
        if has.any():
            u = _uniform(customer_id, salt=0x05057E01, global_seed=seed)
            out = s.to_numpy().astype(object)
            # Заполняем пропуски сгенерированными значениями.
            for ot in _OS_TYPES:
                mask = (~has.to_numpy()) & (os_type == ot)
                if mask.any():
                    pool = _OS_VERSIONS[ot]
                    out[mask] = _pick(u[mask], pool)
            return out
    u = _uniform(customer_id, salt=0x05057E01, global_seed=seed)
    out = np.empty(len(customer_id), dtype=object)
    for ot in _OS_TYPES:
        mask = os_type == ot
        if mask.any():
            out[mask] = _pick(u[mask], _OS_VERSIONS[ot])
    return out


def _gen_browser_name(customer_id, seed) -> np.ndarray:
    u = _uniform(customer_id, salt=0xB6011A11, global_seed=seed)
    probs = np.array([[0.55, 0.10, 0.13, 0.10, 0.05, 0.07]])
    return _categorical(u, np.repeat(probs, len(customer_id), axis=0), _BROWSER_NAMES)


def _gen_browser_version(customer_id, browser_name, seed) -> np.ndarray:
    u = _uniform(customer_id, salt=0xB6011A12, global_seed=seed)
    out = np.empty(len(customer_id), dtype=object)
    for bn in _BROWSER_NAMES:
        mask = browser_name == bn
        if mask.any():
            out[mask] = _pick(u[mask], _BROWSER_VERSIONS[bn])
    return out


def _gen_user_agent(os_type, os_version, browser_name, browser_version) -> np.ndarray:
    out = np.empty(len(os_type), dtype=object)
    for i in range(len(os_type)):
        ot = os_type[i]
        ov = os_version[i]
        bn = browser_name[i]
        bv = browser_version[i]
        if ot == "Windows":
            plat = f"Windows NT 10.0; Win64; x64"
        elif ot == "macOS":
            plat = f"Macintosh; Intel Mac OS X 10_15_7"
        elif ot == "Linux":
            plat = "X11; Linux x86_64"
        elif ot == "Android":
            plat = f"Linux; Android {ov}"
        elif ot == "iOS":
            plat = f"iPhone; CPU iPhone OS {str(ov).replace('.', '_')} like Mac OS X"
        else:
            plat = "Unknown"
        if bn in {"Chrome", "Edge", "Opera", "Yandex"}:
            ua = f"Mozilla/5.0 ({plat}) AppleWebKit/537.36 (KHTML, like Gecko) {bn}/{bv} Safari/537.36"
        elif bn == "Safari":
            ua = f"Mozilla/5.0 ({plat}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{bv} Safari/605.1.15"
        elif bn == "Firefox":
            ua = f"Mozilla/5.0 ({plat}; rv:{bv}) Gecko/20100101 Firefox/{bv}"
        else:
            ua = f"Mozilla/5.0 ({plat}) {bn}/{bv}"
        out[i] = ua
    return out


def _gen_browser_fingerprint(customer_id, browser_name, os_type, seed) -> np.ndarray:
    # Стабильный hex per (customer, browser, os).
    bn_hash = np.array([hash(b) & 0xFFFF for b in browser_name], dtype=np.uint64)
    ot_hash = np.array([hash(o) & 0xFFFF for o in os_type], dtype=np.uint64)
    key = _to_uint64(customer_id) ^ (bn_hash << np.uint64(16)) ^ (ot_hash << np.uint64(32))
    h = _mix(_mix(key ^ np.uint64(0xF1A6E2010BAD7501)) ^ np.uint64(seed))
    return np.array([f"{v:016x}" for v in h.tolist()], dtype=object)


def _gen_canvas_fingerprint(customer_id, browser_name, seed) -> np.ndarray:
    return _hex_id(_to_uint64(customer_id) ^ np.array([hash(b) & 0xFFFFFFFF for b in browser_name], dtype=np.uint64),
                   salt=0xCA10A5F1, global_seed=seed, length=16)


def _gen_audio_fingerprint(customer_id, seed) -> np.ndarray:
    return _hex_id(customer_id, salt=0xA0D10F11, global_seed=seed, length=12)


def _gen_webgl_vendor(customer_id, os_type, seed) -> np.ndarray:
    u = _uniform(customer_id, salt=0x6B9100E3, global_seed=seed)
    out = np.empty(len(customer_id), dtype=object)
    apple_pool = np.array(["Apple Inc."])
    mobile_pool = np.array(["Qualcomm", "Apple Inc.", "Google Inc. (Intel)"])
    desktop_pool = _WEBGL_VENDORS
    out[:] = _pick(u, desktop_pool)
    mask_mobile = (os_type == "Android") | (os_type == "iOS")
    if mask_mobile.any():
        out[mask_mobile] = _pick(u[mask_mobile], mobile_pool)
    mask_apple = os_type == "macOS"
    if mask_apple.any():
        out[mask_apple] = _pick(u[mask_apple], apple_pool)
    return out


# ---------------------------------------------------------------------------
# Screen / language / temporal
# ---------------------------------------------------------------------------

def _parse_screen_resolution(source_screen, customer_id, os_type, seed) -> np.ndarray:
    if source_screen is not None:
        s = source_screen.astype(str).fillna("")
        has = s.str.contains(r"\d+x\d+", regex=True, na=False)
    else:
        s = pd.Series([""] * len(customer_id))
        has = pd.Series([False] * len(customer_id))
    u = _uniform(customer_id, salt=0x5C5EE21E, global_seed=seed)
    out = np.empty(len(customer_id), dtype=object)
    out[:] = ""
    if has.any():
        out[has.to_numpy()] = s[has].to_numpy()
    mask_missing = ~has.to_numpy()
    if mask_missing.any():
        mobile = (os_type == "Android") | (os_type == "iOS")
        m_mob = mask_missing & mobile
        m_desk = mask_missing & ~mobile
        if m_mob.any():
            out[m_mob] = _pick(u[m_mob], _SCREEN_RES_MOBILE)
        if m_desk.any():
            out[m_desk] = _pick(u[m_desk], _SCREEN_RES_DESKTOP)
    return out


def _gen_color_depth(customer_id, seed) -> np.ndarray:
    u = _uniform(customer_id, salt=0xC010D1E1, global_seed=seed)
    return _pick(u, _COLOR_DEPTHS).astype(np.int16)


def _gen_timezone_offset(customer_id, source_timezone, seed) -> np.ndarray:
    if source_timezone is not None:
        v = pd.to_numeric(source_timezone, errors="coerce")
        if v.notna().any():
            return v.fillna(180).astype(np.int32).to_numpy()
    u = _uniform(customer_id, salt=0x721E20FF, global_seed=seed)
    return ((u * 24 - 12).astype(np.int32) * 60)


def _gen_system_language(customer_id, source_lang, seed) -> np.ndarray:
    if source_lang is not None:
        s = source_lang.astype(str).fillna("")
        has = s.str.len() > 1
        if has.any():
            u = _uniform(customer_id, salt=0x5755A6A6, global_seed=seed)
            out = s.to_numpy().astype(object)
            miss = ~has.to_numpy()
            if miss.any():
                out[miss] = _pick(u[miss], _SYSTEM_LANGS)
            return out
    u = _uniform(customer_id, salt=0x5755A6A6, global_seed=seed)
    return _pick(u, _SYSTEM_LANGS).astype(object)


def _gen_temporal(event_dttm) -> tuple[np.ndarray, np.ndarray]:
    dt = pd.to_datetime(event_dttm, errors="coerce", utc=False)
    hour = dt.dt.hour.fillna(12).astype(np.int8).to_numpy()
    dow = dt.dt.dayofweek.fillna(0).astype(np.int8).to_numpy()
    return hour, dow


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

def _gen_merchant_name(event_id, seed) -> np.ndarray:
    u = _uniform(event_id, salt=0x3E6CA42E, global_seed=seed)
    return _pick(u, _MERCHANTS).astype(object)


def _gen_transaction_type(event_id, risk, seed) -> np.ndarray:
    u = _uniform(event_id, salt=0x77A45AC7, global_seed=seed)
    base = np.array([0.55, 0.18, 0.15, 0.08, 0.04])
    fraud = np.array([0.15, 0.35, 0.30, 0.15, 0.05])
    probs = base[None, :] * (1 - risk[:, None]) + fraud[None, :] * risk[:, None]
    return _categorical(u, probs, _TRANSACTION_TYPES)


# ---------------------------------------------------------------------------
# Security flags (binary)
# ---------------------------------------------------------------------------

def _gen_binary(keys, salt, risk, seed, p_benign, p_fraud) -> np.ndarray:
    u = _uniform(keys, salt=salt, global_seed=seed)
    p = _lerp(p_benign, p_fraud, risk)
    return (u < np.clip(p, 0.0, 1.0)).astype(np.int8)


def _gen_is_developer_tools(customer_id, source_dev, risk, seed) -> np.ndarray:
    base = _gen_binary(customer_id, 0xDE7E10F5, risk, seed, 0.02, 0.30)
    if source_dev is not None and len(source_dev) == len(base):
        src = _truthy(source_dev).astype(np.int8)
        return np.maximum(base, src)
    return base


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _gen_ip_address_hash(session_id, seed) -> np.ndarray:
    return _hex_id(session_id, salt=0x19A4D255, global_seed=seed, length=16)


def _gen_connection_type(session_id, os_type, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0xC0117EC7, global_seed=seed)
    out = np.empty(len(session_id), dtype=object)
    mobile_probs = np.array([[0.20, 0.30, 0.25, 0.15, 0.05, 0.05]])
    desktop_probs = np.array([[0.55, 0.05, 0.30, 0.02, 0.03, 0.05]])
    mobile = (os_type == "Android") | (os_type == "iOS")
    if mobile.any():
        out[mobile] = _categorical(
            u[mobile], np.repeat(mobile_probs, mobile.sum(), axis=0), _CONNECTION_TYPES
        )
    desk = ~mobile
    if desk.any():
        out[desk] = _categorical(
            u[desk], np.repeat(desktop_probs, desk.sum(), axis=0), _CONNECTION_TYPES
        )
    return out


def _gen_network_rtt_avg(session_id, risk, rdp, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0x47B12A11, global_seed=seed)
    mean = _lerp(25.0, 220.0, risk) + 60.0 * rdp
    return _exp_sample(u, mean).astype(np.float32)


def _gen_asn_and_isp(customer_id, risk, seed) -> tuple[np.ndarray, np.ndarray]:
    u_foreign = _uniform(customer_id, salt=0xA5114F01, global_seed=seed)
    u_pick = _uniform(customer_id, salt=0xA5114F02, global_seed=seed)
    is_foreign = u_foreign < _lerp(0.03, 0.45, risk)

    ru_idx = np.clip(np.floor(u_pick * len(_ISP_RU)).astype(np.int64), 0, len(_ISP_RU) - 1)
    foreign_idx = np.clip(np.floor(u_pick * len(_ISP_FOREIGN)).astype(np.int64),
                          0, len(_ISP_FOREIGN) - 1)
    asn = np.where(is_foreign, _ASN_FOREIGN[foreign_idx], _ASN_RU[ru_idx]).astype(np.int32)
    isp = np.where(is_foreign, _ISP_FOREIGN[foreign_idx], _ISP_RU[ru_idx]).astype(object)
    return asn, isp


# ---------------------------------------------------------------------------
# Mouse / keyboard / browser behaviour
# ---------------------------------------------------------------------------

def _gen_mouse(session_id, risk, seed) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns velocity_avg, acceleration_avg, jitter_score, linearity_score."""
    u_v = _uniform(session_id, salt=0xA0050E01, global_seed=seed)
    u_a = _uniform(session_id, salt=0xA0050E02, global_seed=seed)
    u_j = _uniform(session_id, salt=0xA0050E03, global_seed=seed)
    u_l = _uniform(session_id, salt=0xA0050E04, global_seed=seed)
    u_bot = _uniform(session_id, salt=0xA0050E00, global_seed=seed)
    is_bot = u_bot < risk * 0.7

    vel_h = 300.0 + u_v * 700.0
    vel_b = 800.0 + u_v * 2500.0
    velocity = np.where(is_bot, vel_b, vel_h).astype(np.float32)

    acc_h = 150.0 + u_a * 350.0
    acc_b = 400.0 + u_a * 1200.0
    acceleration = np.where(is_bot, acc_b, acc_h).astype(np.float32)

    # Jitter: бот → почти 0, человек → 0.4-0.95
    jit_h = 0.4 + u_j * 0.55
    jit_b = 0.0 + u_j * 0.05
    jitter = np.where(is_bot, jit_b, jit_h).astype(np.float32)

    # Linearity: бот → ~1.0 (прямые линии), человек → 0.3-0.8
    lin_h = 0.3 + u_l * 0.5
    lin_b = 0.92 + u_l * 0.08
    linearity = np.where(is_bot, lin_b, lin_h).astype(np.float32)
    return velocity, acceleration, jitter, linearity


def _gen_click_metrics(session_id, risk, seed) -> tuple[np.ndarray, np.ndarray]:
    """click_duration_avg_ms, right_click_count."""
    u_d = _uniform(session_id, salt=0xC11C0001, global_seed=seed)
    u_r = _uniform(session_id, salt=0xC11C0002, global_seed=seed)
    u_bot = _uniform(session_id, salt=0xC11C0000, global_seed=seed)
    is_bot = u_bot < risk * 0.6

    dur_h = 80.0 + u_d * 180.0
    dur_b = 5.0 + u_d * 25.0
    duration = np.where(is_bot, dur_b, dur_h).astype(np.float32)
    # Right clicks: повышены при developer/recon
    mean_rc = _lerp(0.3, 2.0, risk)
    right = np.floor(_exp_sample(u_r, mean_rc)).astype(np.int32)
    return duration, right


def _gen_scroll_velocity(session_id, risk, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0x5C201101, global_seed=seed)
    u_bot = _uniform(session_id, salt=0x5C201100, global_seed=seed)
    is_bot = u_bot < risk * 0.6
    vel_h = 200.0 + u * 800.0
    vel_b = 1500.0 + u * 3500.0
    return np.where(is_bot, vel_b, vel_h).astype(np.float32)


def _gen_keyboard_metrics(session_id, risk, seed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_m = _uniform(session_id, salt=0xCB001501, global_seed=seed)
    u_s = _uniform(session_id, salt=0xCB001502, global_seed=seed)
    u_bot = _uniform(session_id, salt=0xCB001500, global_seed=seed)
    is_bot = u_bot < risk * 0.7

    med_h = 120.0 + u_m * 200.0
    med_b = 30.0 + u_m * 30.0
    median_ms = np.where(is_bot, med_b, med_h).astype(np.float32)
    std_h = 30.0 + u_s * 70.0
    std_b = 1.0 + u_s * 4.0
    std_dev = np.where(is_bot, std_b, std_h).astype(np.float32)
    cv = (std_dev / np.maximum(median_ms, 1.0)).astype(np.float32)
    return median_ms, std_dev, cv


def _gen_clipboard_paste_ratio(session_id, risk, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0xC1B0A57E, global_seed=seed)
    centre = _lerp(0.10, 0.70, risk)
    return np.clip(centre + (u - 0.5) * 0.40, 0.0, 1.0).astype(np.float32)


def _gen_backspace_ratio(session_id, risk, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0xBAC759A5, global_seed=seed)
    centre = _lerp(0.18, 0.04, risk)
    return np.clip(centre + (u - 0.5) * 0.12, 0.0, 1.0).astype(np.float32)


def _gen_copy_paste_counts(session_id, paste_ratio, risk, seed) -> tuple[np.ndarray, np.ndarray]:
    u_c = _uniform(session_id, salt=0xC0901001, global_seed=seed)
    u_p = _uniform(session_id, salt=0xC0901002, global_seed=seed)
    paste_mean = paste_ratio * 8.0 + 0.3
    copy_mean = _lerp(2.0, 0.5, risk)
    copies = np.floor(_exp_sample(u_c, copy_mean)).astype(np.int32)
    pastes = np.floor(_exp_sample(u_p, paste_mean)).astype(np.int32)
    return copies, pastes


def _gen_tab_switch(session_id, risk, rdp, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0xA110B5E1, global_seed=seed)
    mean = _lerp(0.5, 4.0, risk) + 2.0 * rdp
    return np.floor(_exp_sample(u, mean)).astype(np.int32)


def _gen_focus_blur(session_id, risk, rdp, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0xF0CB001B, global_seed=seed)
    mean = _lerp(1.0, 6.0, risk) + 2.5 * rdp
    return np.floor(_exp_sample(u, mean)).astype(np.int32)


def _gen_form_fill_duration(session_id, risk, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0xF02DF111, global_seed=seed)
    mean = _lerp(80.0, 8.0, risk)
    return _exp_sample(u, mean).astype(np.float32)


def _gen_idle_before_submit(session_id, risk, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0x1D1E0001, global_seed=seed)
    mean = _lerp(12.0, 1.0, risk)
    return _exp_sample(u, mean).astype(np.float32)


def _gen_error_correction(session_id, backspace, risk, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0xE6C0BBE7, global_seed=seed)
    base = backspace * 0.8 + (u - 0.5) * 0.05
    base = base - 0.05 * risk
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _gen_hover_time(session_id, risk, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0x40E2A001, global_seed=seed)
    mean = _lerp(220.0, 30.0, risk)
    return _exp_sample(u, mean).astype(np.float32)


def _gen_double_click(session_id, risk, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0xD0BB1CC1, global_seed=seed)
    mean = _lerp(1.5, 0.3, risk)
    return np.floor(_exp_sample(u, mean)).astype(np.int32)


def _gen_drag_drop(session_id, risk, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0xD9A6D9A6, global_seed=seed)
    mean = _lerp(1.0, 0.2, risk)
    return np.floor(_exp_sample(u, mean)).astype(np.int32)


def _gen_resize_events(session_id, risk, rdp, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0x9E512EE1, global_seed=seed)
    mean = _lerp(0.3, 2.5, risk) + 1.5 * rdp
    return np.floor(_exp_sample(u, mean)).astype(np.int32)


def _gen_zoom_level(session_id, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0x200180EE, global_seed=seed)
    # Распределение 0.75-1.5 с пиком на 1.0
    levels = np.array([0.75, 0.9, 1.0, 1.0, 1.0, 1.1, 1.25, 1.5])
    return _pick(u, levels).astype(np.float32)


def _gen_session_duration(session_id, form_dur, idle, risk, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0x55ED0001, global_seed=seed)
    base = form_dur + idle + _lerp(60.0, 10.0, risk) * u + 20.0
    return base.astype(np.float32)


def _gen_pages_visited(session_id, risk, seed) -> np.ndarray:
    u = _uniform(session_id, salt=0xBA9E5001, global_seed=seed)
    mean = _lerp(5.0, 1.5, risk)
    return np.clip(np.floor(_exp_sample(u, mean)).astype(np.int32) + 1, 1, None)


# ---------------------------------------------------------------------------
# Fonts / language
# ---------------------------------------------------------------------------

def _gen_installed_fonts(customer_id, risk, seed) -> np.ndarray:
    u = _uniform(customer_id, salt=0xF0A75001, global_seed=seed)
    centre = _lerp(150.0, 40.0, risk)
    return np.clip(centre + (u - 0.5) * 80.0, 5.0, 600.0).astype(np.int32)


# ---------------------------------------------------------------------------
# Login / device trust
# ---------------------------------------------------------------------------

def _gen_login_method(customer_id, risk, seed) -> np.ndarray:
    u = _uniform(customer_id, salt=0x10610001, global_seed=seed)
    base = np.array([0.40, 0.45, 0.12, 0.03])  # pwd / bio / otp / sso
    fraud = np.array([0.70, 0.10, 0.10, 0.10])
    probs = base[None, :] * (1 - risk[:, None]) + fraud[None, :] * risk[:, None]
    return _categorical(u, probs, _LOGIN_METHODS)


def _gen_failed_login(customer_id, event_id, risk, seed) -> np.ndarray:
    u = _uniform(np.bitwise_xor(_to_uint64(customer_id), _to_uint64(event_id) >> np.uint64(4)),
                 salt=0xFA110610, global_seed=seed)
    mean = _lerp(0.1, 2.0, risk)
    return np.floor(_exp_sample(u, mean)).astype(np.int32)


def _gen_time_since_last_login(customer_id, event_id, risk, seed) -> np.ndarray:
    u = _uniform(np.bitwise_xor(_to_uint64(customer_id), _to_uint64(event_id) >> np.uint64(7)),
                 salt=0x715E0001, global_seed=seed)
    mean = _lerp(86400.0, 120.0, risk)  # норма ~сутки, фрод — минуты
    return _exp_sample(u, mean).astype(np.float32)


def _gen_is_new_device(customer_id, event_id, risk, seed) -> np.ndarray:
    u = _uniform(np.bitwise_xor(_to_uint64(customer_id), _to_uint64(event_id)),
                 salt=0xDE71CE17, global_seed=seed)
    p = _lerp(0.04, 0.55, risk)
    return (u < p).astype(np.int8)


def _gen_is_new_browser(customer_id, event_id, risk, seed) -> np.ndarray:
    u = _uniform(np.bitwise_xor(_to_uint64(customer_id), _to_uint64(event_id) << np.uint64(1)),
                 salt=0xB60B6B60, global_seed=seed)
    p = _lerp(0.05, 0.55, risk)
    return (u < p).astype(np.int8)


def _gen_device_trust(customer_id, risk, is_new_device, seed) -> np.ndarray:
    u = _uniform(customer_id, salt=0x77A7C0001, global_seed=seed)
    base = _lerp(0.85, 0.20, risk) + (u - 0.5) * 0.15
    base = base - 0.15 * is_new_device
    return np.clip(base, 0.0, 1.0).astype(np.float32)


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

    rdp = _truthy(df.get("web_rdp_connection"))
    if len(rdp) != n:
        rdp = np.zeros(n)
    risk = compute_risk_score(df, target)

    out = pd.DataFrame(index=df.index)

    # Identity (passthrough from source).
    out["customer_id"] = pd.Series(customer_id).astype("int64").to_numpy()
    out["event_id"] = pd.Series(event_id).astype("int64").to_numpy()
    out["session_id"] = pd.Series(session_id).astype("int64").to_numpy()

    # OS / Browser identity.
    os_type = _gen_os_type(customer_id, df.get("operating_system_type"), seed)
    os_version = _gen_os_version(customer_id, os_type, df.get("device_system_version"), seed)
    browser_name = _gen_browser_name(customer_id, seed)
    browser_version = _gen_browser_version(customer_id, browser_name, seed)
    out["browser_fingerprint"] = _gen_browser_fingerprint(customer_id, browser_name, os_type, seed)
    out["user_agent"] = _gen_user_agent(os_type, os_version, browser_name, browser_version)
    out["browser_name"] = browser_name
    out["browser_version"] = browser_version
    out["os_type"] = os_type
    out["os_version"] = os_version

    # Temporal.
    if "event_dttm" in df.columns:
        out["event_dttm"] = df["event_dttm"].to_numpy()
        hour, dow = _gen_temporal(df["event_dttm"])
    else:
        out["event_dttm"] = np.array([""] * n, dtype=object)
        hour = np.full(n, 12, dtype=np.int8)
        dow = np.full(n, 0, dtype=np.int8)
    out["hour_of_day"] = hour
    out["day_of_week"] = dow

    # Transaction core (passthrough + synth).
    out["operaton_amt"] = pd.to_numeric(df.get("operaton_amt", 0.0), errors="coerce").fillna(0.0).astype(np.float64).to_numpy()
    out["currency_iso_cd"] = pd.to_numeric(df.get("currency_iso_cd", 643), errors="coerce").fillna(643).astype(np.int32).to_numpy()
    if "mcc_code" in df.columns:
        out["mcc_code"] = df["mcc_code"].astype(str).fillna("0000").to_numpy()
    else:
        out["mcc_code"] = np.array(["0000"] * n, dtype=object)
    out["merchant_name"] = _gen_merchant_name(event_id, seed)
    out["pos_cd"] = pd.to_numeric(df.get("pos_cd", 0), errors="coerce").fillna(0).astype(np.int32).to_numpy()
    out["transaction_type"] = _gen_transaction_type(event_id, risk, seed)

    # Browser flags.
    out["is_developer_tools"] = _gen_is_developer_tools(customer_id, df.get("developer_tools"), risk, seed)
    out["is_headless_browser"] = _gen_binary(session_id, 0x4EAD1E55, risk, seed, 0.005, 0.30)

    # Screen / language.
    out["screen_resolution"] = _parse_screen_resolution(df.get("screen_size"), customer_id, os_type, seed)
    out["screen_color_depth"] = _gen_color_depth(customer_id, seed)
    out["timezone_offset"] = _gen_timezone_offset(customer_id, df.get("timezone"), seed)
    out["system_language"] = _gen_system_language(customer_id, df.get("accept_language"), seed)
    if "browser_language" in df.columns:
        out["browser_language"] = df["browser_language"].astype(str).fillna("ru-RU").to_numpy()
    else:
        out["browser_language"] = _gen_system_language(customer_id, None, seed)
    if "accept_language" in df.columns:
        out["accept_language"] = df["accept_language"].astype(str).fillna("ru-RU").to_numpy()
    else:
        out["accept_language"] = _gen_system_language(customer_id, None, seed)

    # Privacy / fonts.
    out["is_incognito"] = _gen_binary(session_id, 0x10C09017, risk, seed, 0.05, 0.45)
    out["installed_fonts_count"] = _gen_installed_fonts(customer_id, risk, seed)

    # Network.
    out["ip_address_hash"] = _gen_ip_address_hash(session_id, seed)
    out["is_vpn_detected"] = _gen_binary(session_id, 0xF6E700D1, risk, seed, 0.02, 0.40)
    out["is_proxy_detected"] = _gen_binary(session_id, 0xF6E700D2, risk, seed, 0.01, 0.25)
    out["is_tor_detected"] = _gen_binary(session_id, 0xF6E700D3, risk, seed, 0.002, 0.10)
    out["connection_type"] = _gen_connection_type(session_id, os_type, seed)
    out["network_rtt_avg_ms"] = _gen_network_rtt_avg(session_id, risk, rdp, seed)
    asn, isp = _gen_asn_and_isp(customer_id, risk, seed)
    out["asn"] = asn
    out["isp_name"] = isp

    # Mouse biometrics.
    mv, ma, mj, ml = _gen_mouse(session_id, risk, seed)
    out["mouse_velocity_avg"] = mv
    out["mouse_acceleration_avg"] = ma
    out["mouse_jitter_score"] = mj
    out["mouse_linearity_score"] = ml

    # Clicks / scroll.
    cd, rc = _gen_click_metrics(session_id, risk, seed)
    out["click_duration_avg_ms"] = cd
    out["right_click_count"] = rc
    out["scroll_velocity_avg"] = _gen_scroll_velocity(session_id, risk, seed)

    # Keyboard.
    kmed, kstd, kcv = _gen_keyboard_metrics(session_id, risk, seed)
    out["keyboard_typing_speed_median_ms"] = kmed
    out["keyboard_typing_speed_std_dev"] = kstd
    out["keyboard_typing_rhythm_cv"] = kcv

    # Clipboard / backspace.
    out["backspace_ratio"] = _gen_backspace_ratio(session_id, risk, seed)
    paste_ratio = _gen_clipboard_paste_ratio(session_id, risk, seed)
    out["clipboard_paste_ratio"] = paste_ratio
    copies, pastes = _gen_copy_paste_counts(session_id, paste_ratio, risk, seed)
    out["copy_events_count"] = copies
    out["paste_events_count"] = pastes

    # Browser interactions.
    out["tab_switch_count"] = _gen_tab_switch(session_id, risk, rdp, seed)
    out["focus_blur_count"] = _gen_focus_blur(session_id, risk, rdp, seed)
    form_dur = _gen_form_fill_duration(session_id, risk, seed)
    idle = _gen_idle_before_submit(session_id, risk, seed)
    out["form_fill_duration_sec"] = form_dur
    out["idle_time_before_submit_sec"] = idle
    out["error_correction_ratio"] = _gen_error_correction(session_id, out["backspace_ratio"].to_numpy(), risk, seed)
    out["hover_time_avg_ms"] = _gen_hover_time(session_id, risk, seed)
    out["double_click_count"] = _gen_double_click(session_id, risk, seed)
    out["drag_drop_events"] = _gen_drag_drop(session_id, risk, seed)
    out["resize_events_count"] = _gen_resize_events(session_id, risk, rdp, seed)
    out["zoom_level"] = _gen_zoom_level(session_id, seed)

    # Fingerprints.
    out["webgl_vendor"] = _gen_webgl_vendor(customer_id, os_type, seed)
    out["canvas_fingerprint"] = _gen_canvas_fingerprint(customer_id, browser_name, seed)
    out["audio_fingerprint"] = _gen_audio_fingerprint(customer_id, seed)

    # Session shape.
    out["session_duration_sec"] = _gen_session_duration(session_id, form_dur, idle, risk, seed)
    out["pages_visited_count"] = _gen_pages_visited(session_id, risk, seed)

    # Login / trust.
    out["login_method"] = _gen_login_method(customer_id, risk, seed)
    out["failed_login_attempts"] = _gen_failed_login(customer_id, event_id, risk, seed)
    out["time_since_last_login_sec"] = _gen_time_since_last_login(customer_id, event_id, risk, seed)
    is_new_dev = _gen_is_new_device(customer_id, event_id, risk, seed)
    out["is_new_device"] = is_new_dev
    out["is_new_browser"] = _gen_is_new_browser(customer_id, event_id, risk, seed)
    out["device_trust_score"] = _gen_device_trust(customer_id, risk, is_new_dev, seed)

    return out[TASK_COLUMNS]
