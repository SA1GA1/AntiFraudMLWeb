"""Environment-driven configuration for the daily flow.

All paths and endpoints are overridable via env vars so the flow can be
moved between hosts without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FlowConfig:
    events_glob: str
    labels_glob: str
    state_path: Path
    work_dir: Path          # where aggregate/extract write parquet outputs
    output_dir: Path        # where train writes best.pt + metrics.json
    mlflow_uri: str
    mlflow_experiment: str
    model_name: str
    backend_reload_url: str
    backend_reload_token: str | None
    auc_tolerance: float
    epochs: int
    label_lag_days: int     # newest label cut-off (today - lag_days)
    label_window_days: int  # window length (lag_days .. lag_days + window)

    @classmethod
    def from_env(cls) -> "FlowConfig":
        root = Path(os.environ.get("FRAUD_ROOT", str(Path.home() / "fraud")))
        return cls(
            events_glob=os.environ.get(
                "FRAUD_EVENTS_GLOB", str(root / "events" / "dt=*" / "*.parquet")
            ),
            labels_glob=os.environ.get(
                "FRAUD_LABELS_GLOB", str(root / "labels" / "dt=*" / "*.parquet")
            ),
            state_path=Path(os.environ.get(
                "FRAUD_STATE_PATH",
                str(root / "state" / "customer_features.state.parquet"),
            )),
            work_dir=Path(os.environ.get(
                "FRAUD_WORK_DIR", str(root / "work"),
            )),
            output_dir=Path(os.environ.get(
                "FRAUD_OUTPUT_DIR", "trainer/checkpoints"
            )),
            mlflow_uri=os.environ.get(
                "MLFLOW_TRACKING_URI", f"file://{root / 'mlruns'}"
            ),
            mlflow_experiment=os.environ.get(
                "MLFLOW_EXPERIMENT", "fraud_mlp_web"
            ),
            model_name=os.environ.get(
                "FRAUD_MODEL_NAME", "fraud_mlp_web"
            ),
            backend_reload_url=os.environ.get(
                "FRAUD_BACKEND_RELOAD_URL",
                "http://localhost:8000/admin/reload-model",
            ),
            backend_reload_token=os.environ.get("FRAUD_BACKEND_RELOAD_TOKEN"),
            auc_tolerance=float(os.environ.get("FRAUD_AUC_TOLERANCE", "0.005")),
            epochs=int(os.environ.get("FRAUD_EPOCHS", "20")),
            label_lag_days=int(os.environ.get("FRAUD_LABEL_LAG_DAYS", "7")),
            label_window_days=int(os.environ.get("FRAUD_LABEL_WINDOW_DAYS", "30")),
        )
