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

    print("\n[main] для предсказания по одному JSON-событию см. predict_from_json.py")
    return
    # ----- Старые hardcoded демо удалены, оставлено для исторической ссылки -----
    suspicious_event = {
        "customer_id": 123123123123129,           # клиент с историей
        "event_id": 999999999999999,
        "event_dttm": "2025-08-09 03:45:00",      # ночное время
        "event_type_nm": 1,
        "event_desc": 42,
        "channel_indicator_type": 2,
        "channel_indicator_sub_type": 1,
        "operaton_amt": 250000.0,                  # крупная сумма
        "currency_iso_cd": 643,
        "mcc_code": "6011",
        "pos_cd": 0,
        "accept_language": "en-US",
        "browser_language": "en-US",
        "timezone": 180,
        "session_id": 125_000_000_000_000,
        "operating_system_type": 1,
        "battery": "0.95",
        "device_system_version": "16.0",
        "screen_size": "1170x2532",
        "developer_tools": "1",                    # ⚠ developer tools on
        "phone_voip_call_state": 1,                # ⚠ VoIP-звонок
        "web_rdp_connection": 1,                   # ⚠ RDP
        "compromised": "1",                        # ⚠ root
        # синтетические признаки — типичные для фрода:
        "attestation_status": "failed_root",
        "app_background_events": 8,
        "clipboard_paste_ratio_mobile": 0.7,
        "entry_source": "sms_link",
        "connection_type": "vpn",
        "touch_typing_rhythm": 0.06,
        "sim_country_mismatch": 1,
        "network_rtt_avg": 280.0,
        "biometric_entry_used": 0,
        "storage_free_percent": 8.0,
        "battery_charging_state": "plugged_24_7",
        "screen_orientation_changes": 12,
        "debugger_attached": 1,
    }
    out = single_predict(model, preproc, suspicious_event, agg_df, device=device)
    print(f"  logit={out['logit']:+.3f}  P(fraud)={out['fraud_probability']:.4f}  has_history={out['has_history']}")

    print("\n[main] пример «нормального» события:")
    benign_event = {
        **suspicious_event,
        "event_dttm": "2025-08-09 14:30:00",
        "operaton_amt": 1500.0,
        "developer_tools": "0",
        "phone_voip_call_state": 0,
        "web_rdp_connection": 0,
        "compromised": "0",
        "attestation_status": "passed",
        "app_background_events": 0,
        "clipboard_paste_ratio_mobile": 0.05,
        "entry_source": "manual",
        "connection_type": "wifi_home",
        "touch_typing_rhythm": 0.45,
        "sim_country_mismatch": 0,
        "network_rtt_avg": 25.0,
        "biometric_entry_used": 1,
        "storage_free_percent": 72.0,
        "battery_charging_state": "discharging",
        "screen_orientation_changes": 0,
        "debugger_attached": 0,
    }
    out = single_predict(model, preproc, benign_event, agg_df, device=device)
    print(f"  logit={out['logit']:+.3f}  P(fraud)={out['fraud_probability']:.4f}  has_history={out['has_history']}")


if __name__ == "__main__":
    main()
