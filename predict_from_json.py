"""[DEPRECATED — dev/debug only]  JSON-инференс из чекпоинта.

Production-инференс уехал в AntiFraudMain (``mlflow.pyfunc.load_model``).
Этот скрипт оставлен как dev-утилита для разработки фичей и быстрой
проверки чекпоинта на готовом JSON-payload'е.

Запуск:
    python3 predict_from_json.py predict_input_example.json
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch

from trainer.model import FraudMLP


def main(json_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load("trainer/checkpoints/best.pt", map_location=device, weights_only=False)
    preproc = pickle.loads(ck["preprocessor"])
    model = FraudMLP(
        cat_vocab_sizes=ck["cat_vocab_sizes"],
        n_numeric=ck["n_numeric"],
        n_aggregate=ck["n_aggregate"],
    )
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()

    agg_df = pq.read_table("data_augmented/customer_features.parquet").to_pandas()

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    events = [
        {k: v for k, v in e.items() if not k.startswith("_")}
        for e in payload["events"]
    ]
    df = pd.DataFrame(events)

    num, cat = preproc.transform_events(df)
    agg, has_hist = preproc.transform_aggregates(df["customer_id"].to_numpy(), agg_df)

    n_events = len(events)
    predictions = []

    for i in range(n_events):
        # Синхронизация GPU для точного замера времени
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        with torch.no_grad():
            # Срезаем один семпл, сохраняя размерность батча (i:i+1)
            logit = model(
                torch.from_numpy(num[i:i+1]).to(device),
                torch.from_numpy(cat[i:i+1]).to(device),
                torch.from_numpy(agg[i:i+1]).to(device),
                torch.from_numpy(has_hist[i:i+1]).to(device),
            )
        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        prob = torch.sigmoid(logit).item()
        logit_val = logit.item()
        predictions.append((logit_val, prob, elapsed))

    for i, ev in enumerate(payload["events"]):
        logit_val, prob, elapsed = predictions[i]
        label = ev.get("_label", "(без описания)")
        print(
            f"event_id={ev['event_id']}  "
            f"logit={logit_val:+.3f}  "
            f"P(fraud)={prob:.4f}  "
            f"has_history={bool(has_hist[i, 0])}  "
            f"time={elapsed*1000:.2f}ms  "
            f"— {label}"
        )


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("predict_input_example.json")
    main(path)