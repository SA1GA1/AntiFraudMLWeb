"""Пример использования обученной модели FraudMLP.

Запуск:
    python3 predict_example.py

Скрипт показывает два сценария:
  1. Batch-инференс по test.parquet с записью submission.csv
  2. Предсказание для одного вручную построенного события
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from trainer.model import FraudMLP


CHECKPOINT_PATH = Path("trainer/checkpoints/best.pt")
AGGREGATES_PATH = Path("data_augmented/customer_features.parquet")
TEST_PATH = Path("data_augmented/test.parquet")
SUBMISSION_PATH = Path("submission.csv")


# ---------------------------------------------------------------------------
# 1. Загрузка модели и препроцессора
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: str = "cpu"):
    ck = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    preproc = pickle.loads(ck["preprocessor"])
    model = FraudMLP(
        cat_vocab_sizes=ck["cat_vocab_sizes"],
        n_numeric=ck["n_numeric"],
        n_aggregate=ck["n_aggregate"],
    )
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    print(
        f"[load] чекпоинт эпохи {ck['epoch']}, "
        f"val AUC={ck['val_auc']:.4f}, PR-AUC={ck['val_pr_auc']:.4f}"
    )
    return model, preproc


# ---------------------------------------------------------------------------
# 2. Batch-инференс по всему test.parquet → submission.csv
# ---------------------------------------------------------------------------

def batch_predict(
    model: FraudMLP,
    preproc,
    test_path: Path,
    agg_df: pd.DataFrame | None,
    device: str = "cpu",
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Возвращает DataFrame с event_id и predict (raw logit)."""
    pf = pq.ParquetFile(str(test_path))
    total = pf.metadata.num_rows
    print(f"[batch] {test_path.name}: {total:,} строк")

    event_ids: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    written = 0

    with torch.no_grad():
        for record_batch in pf.iter_batches(batch_size=batch_size):
            df = record_batch.to_pandas()
            num, cat = preproc.transform_events(df)
            agg, has_hist = preproc.transform_aggregates(
                df["customer_id"].to_numpy(), agg_df
            )

            num_t = torch.from_numpy(num).to(device)
            cat_t = torch.from_numpy(cat).to(device)
            agg_t = torch.from_numpy(agg).to(device)
            hist_t = torch.from_numpy(has_hist).to(device)

            logits = model(num_t, cat_t, agg_t, hist_t).cpu().numpy()
            event_ids.append(df["event_id"].to_numpy())
            scores.append(logits)

            written += len(df)
            print(f"  ... {written:,}/{total:,}", end="\r")
    print()

    return pd.DataFrame(
        {
            "event_id": np.concatenate(event_ids),
            "predict": np.concatenate(scores).astype(np.float32),
        }
    )


# ---------------------------------------------------------------------------
# 3. Предсказание для одного события (вручную сконструированный пример)
# ---------------------------------------------------------------------------

def single_predict(
    model: FraudMLP,
    preproc,
    event: dict,
    agg_df: pd.DataFrame | None,
    device: str = "cpu",
) -> dict:
    """event — словарь со значениями полей одного события."""
    df = pd.DataFrame([event])
    num, cat = preproc.transform_events(df)
    agg, has_hist = preproc.transform_aggregates(
        df["customer_id"].to_numpy(), agg_df
    )
    with torch.no_grad():
        logits = model(
            torch.from_numpy(num).to(device),
            torch.from_numpy(cat).to(device),
            torch.from_numpy(agg).to(device),
            torch.from_numpy(has_hist).to(device),
        )
        prob = torch.sigmoid(logits).cpu().numpy()[0]
    return {
        "logit": float(logits.cpu().numpy()[0]),
        "fraud_probability": float(prob),
        "has_history": bool(has_hist[0, 0]),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] device: {device}")

    model, preproc = load_model(CHECKPOINT_PATH, device=device)

    agg_df = None
    if AGGREGATES_PATH.exists():
        agg_df = pq.read_table(str(AGGREGATES_PATH)).to_pandas()
        print(f"[main] загружено {len(agg_df):,} клиентских профилей")

    # --- Сценарий 1: batch по test.parquet → submission.csv ---
    if TEST_PATH.exists():
        preds = batch_predict(model, preproc, TEST_PATH, agg_df, device=device)
        preds.to_csv(SUBMISSION_PATH, index=False)
        print(f"[main] записано {len(preds):,} предсказаний -> {SUBMISSION_PATH}")
        print(f"[main] статистика predict: "
              f"min={preds['predict'].min():.3f}  "
              f"mean={preds['predict'].mean():.3f}  "
              f"max={preds['predict'].max():.3f}")
        # топ-5 самых подозрительных событий
        top = preds.nlargest(5, "predict")
        print("\n[main] топ-5 событий с максимальным риском:")
        for _, row in top.iterrows():
            prob = float(torch.sigmoid(torch.tensor(row["predict"])))
            print(f"  event_id={row['event_id']}  logit={row['predict']:+.3f}  P(fraud)={prob:.4f}")
    else:
        print(f"[main] {TEST_PATH} не найден — пропускаю batch")

    print("\n[main] пример прямого вызова single_predict (web-fraud схема task.md):")
    suspicious_event = {
        "customer_id": 123123123123129,
        "event_id": 999999999999999,
        "session_id": 125_000_000_000_000,
        "event_dttm": "2025-08-09 03:45:00",
        "hour_of_day": 3,
        "day_of_week": 5,
        "operaton_amt": 250000.0,
        "currency_iso_cd": 643,
        "mcc_code": "6011",
        "merchant_name": "Tinkoff",
        "pos_cd": 0,
        "transaction_type": "p2p",
        "browser_fingerprint": "deadbeefcafef00d",
        "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0",
        "browser_name": "Chrome",
        "browser_version": "120.0",
        "os_type": "Linux",
        "os_version": "Ubuntu 22.04",
        "is_developer_tools": 1,
        "is_headless_browser": 1,
        "is_incognito": 1,
        "screen_resolution": "1920x1080",
        "screen_color_depth": 24,
        "timezone_offset": 480,
        "system_language": "zh-CN",
        "browser_language": "zh-CN",
        "accept_language": "zh-CN",
        "installed_fonts_count": 20,
        "ip_address_hash": "deadbeef01ab02cd",
        "is_vpn_detected": 1,
        "is_proxy_detected": 1,
        "is_tor_detected": 0,
        "connection_type": "cellular",
        "network_rtt_avg_ms": 280.0,
        "asn": 4134,
        "isp_name": "China Telecom",
        "mouse_velocity_avg": 2200.0,
        "mouse_acceleration_avg": 1100.0,
        "mouse_jitter_score": 0.02,
        "mouse_linearity_score": 0.98,
        "click_duration_avg_ms": 12.0,
        "right_click_count": 4,
        "scroll_velocity_avg": 3500.0,
        "keyboard_typing_speed_median_ms": 35.0,
        "keyboard_typing_speed_std_dev": 2.0,
        "keyboard_typing_rhythm_cv": 0.05,
        "backspace_ratio": 0.01,
        "clipboard_paste_ratio": 0.78,
        "copy_events_count": 0,
        "paste_events_count": 9,
        "tab_switch_count": 6,
        "focus_blur_count": 7,
        "form_fill_duration_sec": 4.0,
        "idle_time_before_submit_sec": 1.0,
        "error_correction_ratio": 0.02,
        "hover_time_avg_ms": 22.0,
        "double_click_count": 0,
        "drag_drop_events": 0,
        "resize_events_count": 3,
        "zoom_level": 1.0,
        "webgl_vendor": "Google Inc. (Intel)",
        "canvas_fingerprint": "abcdef0123456789",
        "audio_fingerprint": "feeddead0123",
        "session_duration_sec": 35.0,
        "pages_visited_count": 2,
        "login_method": "password",
        "failed_login_attempts": 3,
        "time_since_last_login_sec": 60.0,
        "is_new_device": 1,
        "is_new_browser": 1,
        "device_trust_score": 0.12,
    }
    out = single_predict(model, preproc, suspicious_event, agg_df, device=device)
    print(f"  фрод-кейс: logit={out['logit']:+.3f}  P(fraud)={out['fraud_probability']:.4f}  has_history={out['has_history']}")

    benign_event = {
        **suspicious_event,
        "event_dttm": "2025-08-09 14:30:00",
        "hour_of_day": 14,
        "operaton_amt": 1500.0,
        "merchant_name": "Pyaterochka",
        "transaction_type": "payment",
        "is_developer_tools": 0,
        "is_headless_browser": 0,
        "is_incognito": 0,
        "timezone_offset": 180,
        "system_language": "ru-RU",
        "browser_language": "ru-RU",
        "accept_language": "ru-RU",
        "installed_fonts_count": 180,
        "is_vpn_detected": 0,
        "is_proxy_detected": 0,
        "is_tor_detected": 0,
        "connection_type": "wifi",
        "network_rtt_avg_ms": 22.0,
        "asn": 8359,
        "isp_name": "Rostelecom",
        "mouse_velocity_avg": 480.0,
        "mouse_acceleration_avg": 250.0,
        "mouse_jitter_score": 0.62,
        "mouse_linearity_score": 0.55,
        "click_duration_avg_ms": 130.0,
        "right_click_count": 0,
        "scroll_velocity_avg": 450.0,
        "keyboard_typing_speed_median_ms": 220.0,
        "keyboard_typing_speed_std_dev": 65.0,
        "keyboard_typing_rhythm_cv": 0.29,
        "backspace_ratio": 0.18,
        "clipboard_paste_ratio": 0.05,
        "copy_events_count": 1,
        "paste_events_count": 1,
        "tab_switch_count": 0,
        "focus_blur_count": 1,
        "form_fill_duration_sec": 75.0,
        "idle_time_before_submit_sec": 8.0,
        "error_correction_ratio": 0.16,
        "hover_time_avg_ms": 250.0,
        "double_click_count": 1,
        "drag_drop_events": 0,
        "resize_events_count": 0,
        "login_method": "biometric",
        "failed_login_attempts": 0,
        "time_since_last_login_sec": 86400.0,
        "is_new_device": 0,
        "is_new_browser": 0,
        "device_trust_score": 0.92,
    }
    out = single_predict(model, preproc, benign_event, agg_df, device=device)
    print(f"  норма:     logit={out['logit']:+.3f}  P(fraud)={out['fraud_probability']:.4f}  has_history={out['has_history']}")
    print("\n[main] для предсказания по одному JSON-событию см. predict_from_json.py")


if __name__ == "__main__":
    main()
