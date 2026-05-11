"""Prefect daily retrain flow.

Stages
------
1. ``data_hash``  — fingerprint of input partitions (file size + mtime, sorted).
2. ``aggregate``  — incremental update of ``customer_features.parquet`` from
   ``events_glob``, persisting raw sums to ``state_path``.
3. ``extract``    — join events with labels (windowed for chargeback lag) to
   build ``labelled_events.parquet``.
4. ``train``      — train FraudMLP, log MLFlow run + register a new pyfunc
   version of ``fraud_mlp_web``.
5. ``promote``    — gate by val_auc; transition to Production if within
   tolerance of the prior Prod version.
6. ``notify``     — POST AntiFraudMain ``/admin/reload-model`` so the
   inference service picks up the new version.

Prefect is required at flow-definition time. Each task shells out to
``trainer.cli`` rather than calling Python in-process so failures stay
contained and logs land in Prefect's stdout capture.
"""

from __future__ import annotations

import glob as _glob
import hashlib
import subprocess
from datetime import date, timedelta
from pathlib import Path

from prefect import flow, get_run_logger, task

from .config import FlowConfig
from .notify import notify_backend
from .promote import validate_and_promote


@task(retries=2, retry_delay_seconds=30)
def compute_data_hash(events_glob: str) -> str:
    """Stable fingerprint of the input set: sorted (path, size, mtime_int)."""
    h = hashlib.sha256()
    paths = sorted(_glob.glob(events_glob))
    for p in paths:
        st = Path(p).stat()
        h.update(p.encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:16]


@task(retries=3, retry_delay_seconds=[60, 300, 900])
def run_aggregate(events_glob: str, state_path: str) -> None:
    subprocess.run(
        [
            "python", "-m", "trainer.cli", "aggregate",
            "--events-glob", events_glob,
            "--incremental",
            "--state-path", state_path,
        ],
        check=True,
    )


@task(retries=2, retry_delay_seconds=[60, 300])
def run_extract(
    events_glob: str, labels_glob: str, label_window_end: str
) -> None:
    subprocess.run(
        [
            "python", "-m", "trainer.cli", "extract",
            "--events-glob", events_glob,
            "--labels-glob", labels_glob,
            "--label-window-end", label_window_end,
        ],
        check=True,
    )


@task(retries=1, retry_delay_seconds=120)
def run_train(
    mlflow_uri: str,
    mlflow_experiment: str,
    model_name: str,
    data_hash: str,
    epochs: int,
    output_dir: str,
) -> None:
    subprocess.run(
        [
            "python", "-m", "trainer.cli", "train",
            "--mlflow-uri", mlflow_uri,
            "--mlflow-experiment", mlflow_experiment,
            "--registered-model-name", model_name,
            "--data-hash", data_hash,
            "--epochs", str(epochs),
            "--out", output_dir,
        ],
        check=True,
    )


@task
def run_promote(
    model_name: str, mlflow_uri: str, auc_tolerance: float
) -> dict:
    return validate_and_promote(model_name, mlflow_uri, auc_tolerance)


@task(retries=3, retry_delay_seconds=15)
def run_notify(url: str, token: str | None, payload: dict) -> dict:
    return notify_backend(url=url, payload=payload, token=token)


@flow(name="fraud-daily-retrain")
def daily_flow(
    label_window_end: str | None = None,
    epochs: int | None = None,
) -> dict:
    """Entry point used by `orchestration.deployment` and ad-hoc CLI runs."""
    cfg = FlowConfig.from_env()
    logger = get_run_logger()

    # Label window: events from chargeback-aged days only.
    if label_window_end is None:
        label_window_end = (date.today() - timedelta(days=cfg.label_lag_days)).isoformat()
    epochs = epochs if epochs is not None else cfg.epochs

    logger.info("flow start  events=%s  labels=%s  window_end=%s",
                cfg.events_glob, cfg.labels_glob, label_window_end)

    data_hash = compute_data_hash(cfg.events_glob)
    logger.info("data_hash=%s", data_hash)

    agg = run_aggregate.submit(cfg.events_glob, str(cfg.state_path))
    ext = run_extract.submit(
        cfg.events_glob, cfg.labels_glob, label_window_end, wait_for=[agg]
    )
    tr = run_train.submit(
        cfg.mlflow_uri, cfg.mlflow_experiment, cfg.model_name,
        data_hash, epochs, str(cfg.output_dir),
        wait_for=[ext],
    )
    decision = run_promote.submit(
        cfg.model_name, cfg.mlflow_uri, cfg.auc_tolerance, wait_for=[tr]
    ).result()
    logger.info("promote decision: %s", decision)

    if decision.get("promoted"):
        notify_result = run_notify.submit(
            cfg.backend_reload_url,
            cfg.backend_reload_token,
            {"model_name": cfg.model_name, "version": decision.get("version")},
        ).result()
        logger.info("backend reload: %s", notify_result)
        return {"promoted": True, "decision": decision, "notify": notify_result}

    logger.warning("not promoted (%s) — skipping backend reload", decision.get("reason"))
    return {"promoted": False, "decision": decision}


if __name__ == "__main__":
    daily_flow()
