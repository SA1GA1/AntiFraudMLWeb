"""MLFlow PyFunc wrapper for the FraudMLP checkpoint.

Imported only when MLFlow is available — the rest of the trainer must remain
runnable without mlflow installed.

Bundled via `code_paths=["trainer"]` in `mlflow.pyfunc.log_model(...)` so that
`Preprocessor` (pickled inside the checkpoint) and `FraudMLP` resolve at
load-time on the inference side.
"""

from __future__ import annotations

import pickle
from typing import Any

import mlflow.pyfunc  # noqa: F401 — required at module import time
import numpy as np
import pandas as pd
import torch

from .model import FraudMLP


class FraudPyfunc(mlflow.pyfunc.PythonModel):
    """Carries the checkpoint + customer_features parquet as model artifacts.

    `model_input` must be a DataFrame with task.md schema columns (at minimum
    `customer_id` plus whatever event-level columns the Preprocessor expects).
    Missing columns get filled with NaN by the Preprocessor and z-scored to 0.
    """

    def load_context(self, context: Any) -> None:
        ck = torch.load(context.artifacts["checkpoint"], map_location="cpu",
                         weights_only=False)
        self._preproc = pickle.loads(ck["preprocessor"])
        self._model = FraudMLP(
            cat_vocab_sizes=ck["cat_vocab_sizes"],
            n_numeric=ck["n_numeric"],
            n_aggregate=ck["n_aggregate"],
        )
        self._model.load_state_dict(ck["model_state"])
        self._model.eval()

        agg_path = context.artifacts.get("customer_features")
        if agg_path:
            self._agg_df = pd.read_parquet(agg_path)
        else:
            self._agg_df = None

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        num, cat = self._preproc.transform_events(model_input)
        agg, has_hist = self._preproc.transform_aggregates(
            model_input["customer_id"].to_numpy(), self._agg_df
        )
        num_t = torch.from_numpy(num)
        cat_t = torch.from_numpy(cat)
        agg_t = torch.from_numpy(agg)
        hist_t = torch.from_numpy(has_hist)
        with torch.no_grad():
            logits = self._model(num_t, cat_t, agg_t, hist_t).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        return pd.DataFrame({"logit": logits.ravel(), "p_fraud": probs.ravel()})
