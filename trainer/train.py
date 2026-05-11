"""Training loop: load labelled events + customer aggregates, fit preprocessor,
train FraudMLP, save best checkpoint + per-epoch metrics."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import FraudDataset
from .model import FraudMLP
from .preprocess import Preprocessor


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    agg_dropout: float = 0.20
    val_frac: float = 0.20
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # MLFlow integration — все поля опциональные; если mlflow_uri is None,
    # вся трекинг-логика становится no-op'ом и mlflow в рантайме не требуется.
    mlflow_uri: str | None = None
    mlflow_experiment: str = "fraud_mlp_web"
    mlflow_run_name: str | None = None
    registered_model_name: str | None = None  # e.g. "fraud_mlp_web"
    data_hash: str | None = None  # tag, прокидывается из daily_flow / DVC


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC, no sklearn dependency."""
    scores = scores.astype(np.float64)
    labels = labels.astype(np.int8)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # ranks: smallest score → rank 1; ties get average rank
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average-rank for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
        i = j + 1
    sum_pos_ranks = ranks[labels == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """PR-AUC via the average-precision formulation."""
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order].astype(np.float64)
    cum_tp = np.cumsum(labels_sorted)
    precisions = cum_tp / np.arange(1, len(labels) + 1)
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    return float((precisions * labels_sorted).sum() / n_pos)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _stratified_split(target: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(target))
    train_idx, val_idx = [], []
    for cls in (0, 1):
        cls_idx = idx[target == cls]
        rng.shuffle(cls_idx)
        n_val = int(len(cls_idx) * val_frac)
        val_idx.append(cls_idx[:n_val])
        train_idx.append(cls_idx[n_val:])
    return (
        np.concatenate(train_idx),
        np.concatenate(val_idx),
    )


def _materialise(
    df: pd.DataFrame,
    preproc: Preprocessor,
    agg_df: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num, cat = preproc.transform_events(df)
    agg, has_hist = preproc.transform_aggregates(
        df["customer_id"].to_numpy(), agg_df
    )
    return num, cat, agg, has_hist


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


class _Tracker:
    """Thin adapter over mlflow — no-op if config.mlflow_uri is None."""

    def __init__(self, config: TrainConfig) -> None:
        self.enabled = bool(config.mlflow_uri)
        self._mlflow = None
        if not self.enabled:
            return
        try:
            import mlflow  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mlflow_uri set but `mlflow` is not installed. "
                "Either pip install mlflow or drop the --mlflow-uri flag."
            ) from e
        mlflow.set_tracking_uri(config.mlflow_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        self._mlflow = mlflow

    @contextmanager
    def start_run(self, run_name: str | None = None):
        if not self.enabled:
            yield None
            return
        with self._mlflow.start_run(run_name=run_name) as run:
            yield run

    def log_params(self, params: dict) -> None:
        if self.enabled:
            # mlflow требует строки/числа
            safe = {k: ("" if v is None else v) for k, v in params.items()}
            self._mlflow.log_params(safe)

    def set_tag(self, key: str, value: str | None) -> None:
        if self.enabled and value:
            self._mlflow.set_tag(key, value)

    def log_metrics(self, metrics: dict, step: int) -> None:
        if self.enabled:
            clean = {k: float(v) for k, v in metrics.items()
                     if v is not None and np.isfinite(v)}
            if clean:
                self._mlflow.log_metrics(clean, step=step)

    def log_pyfunc(
        self,
        checkpoint_path: Path,
        customer_features_path: Path | None,
        registered_model_name: str | None,
    ) -> str | None:
        if not self.enabled:
            return None
        from .pyfunc import FraudPyfunc  # lazy: mlflow required at import time

        artifacts: dict[str, str] = {"checkpoint": str(checkpoint_path)}
        if customer_features_path and customer_features_path.exists():
            artifacts["customer_features"] = str(customer_features_path)

        trainer_dir = Path(__file__).resolve().parent
        info = self._mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=FraudPyfunc(),
            artifacts=artifacts,
            code_paths=[str(trainer_dir)],
            registered_model_name=registered_model_name,
        )
        return info.model_uri


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    labelled_path: Path,
    aggregates_path: Path | None,
    output_dir: Path,
    config: TrainConfig,
    progress: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    tracker = _Tracker(config)
    run_name = config.mlflow_run_name or (
        f"fraud_mlp_e{config.epochs}_d{config.agg_dropout:.2f}_s{config.seed}"
    )

    with tracker.start_run(run_name=run_name):
        tracker.log_params(asdict(config))
        tracker.set_tag("git_sha", _git_sha())
        tracker.set_tag("data_hash", config.data_hash)
        tracker.set_tag("device", config.device)
        return _train_impl(
            labelled_path=labelled_path,
            aggregates_path=aggregates_path,
            output_dir=output_dir,
            config=config,
            tracker=tracker,
            progress=progress,
        )


def _train_impl(
    labelled_path: Path,
    aggregates_path: Path | None,
    output_dir: Path,
    config: TrainConfig,
    tracker: _Tracker,
    progress: bool,
) -> dict[str, Any]:
    if progress:
        print(f"[train] device: {config.device}")
        print(f"[train] loading labelled events from {labelled_path}")
    labelled = pq.read_table(str(labelled_path)).to_pandas()
    if progress:
        print(f"[train]   {len(labelled):,} rows; target dist: {labelled['target'].value_counts().to_dict()}")

    agg_df = None
    if aggregates_path is not None and aggregates_path.exists():
        if progress:
            print(f"[train] loading aggregates from {aggregates_path}")
        agg_df = pq.read_table(str(aggregates_path)).to_pandas()
        if progress:
            cov = labelled["customer_id"].isin(agg_df["customer_id"]).mean()
            print(f"[train]   {len(agg_df):,} customers; coverage of labelled set: {cov:.4f}")
    else:
        if progress:
            print("[train] no aggregates file — has_history will be 0 for all rows")

    target = labelled["target"].astype(np.int8).to_numpy()
    train_idx, val_idx = _stratified_split(target, config.val_frac, config.seed)
    df_train = labelled.iloc[train_idx].reset_index(drop=True)
    df_val = labelled.iloc[val_idx].reset_index(drop=True)
    if progress:
        print(f"[train] split: train={len(df_train):,}  val={len(df_val):,}")

    if progress:
        print("[train] fitting preprocessor on training split")
    preproc = Preprocessor().fit(df_train, agg_df=agg_df)

    if progress:
        print("[train] materialising tensors")
    num_tr, cat_tr, agg_tr, hist_tr = _materialise(df_train, preproc, agg_df)
    num_va, cat_va, agg_va, hist_va = _materialise(df_val, preproc, agg_df)
    y_tr = df_train["target"].to_numpy().astype(np.float32)
    y_va = df_val["target"].to_numpy().astype(np.float32)

    train_ds = FraudDataset(num_tr, cat_tr, agg_tr, hist_tr, y_tr,
                            train_mode=True, agg_dropout=config.agg_dropout,
                            seed=config.seed)
    val_ds = FraudDataset(num_va, cat_va, agg_va, hist_va, y_va,
                          train_mode=False, agg_dropout=0.0,
                          seed=config.seed)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(config.device == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(config.device == "cuda"),
    )

    model = FraudMLP(
        cat_vocab_sizes=preproc.cat_vocab_sizes,
        n_numeric=preproc.n_numeric,
        n_aggregate=preproc.n_aggregate,
    ).to(config.device)

    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )

    best_auc = -1.0
    best_path = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.json"
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for batch in train_loader:
            num, cat, agg, hist, y = (b.to(config.device, non_blocking=True) for b in batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(num, cat, agg, hist)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(y)
            train_n += len(y)
        scheduler.step()
        train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                num, cat, agg, hist, y = (b.to(config.device, non_blocking=True) for b in batch)
                logits = model(num, cat, agg, hist)
                loss = criterion(logits, y)
                val_loss_sum += float(loss.item()) * len(y)
                val_n += len(y)
                all_logits.append(logits.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        val_loss = val_loss_sum / max(val_n, 1)
        scores = np.concatenate(all_logits)
        labels = np.concatenate(all_targets).astype(np.int8)
        auc = _roc_auc(scores, labels)
        pr_auc = _average_precision(scores, labels)
        dt = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": auc,
            "val_pr_auc": pr_auc,
            "lr": optimizer.param_groups[0]["lr"],
            "secs": dt,
        }
        history.append(entry)
        if progress:
            print(
                f"[train] epoch {epoch:02d}/{config.epochs}  "
                f"loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
                f"AUC {auc:.4f}  PR-AUC {pr_auc:.4f}  "
                f"lr {entry['lr']:.2e}  {dt:.1f}s"
            )

        with open(metrics_path, "w") as f:
            json.dump({"config": asdict(config), "history": history}, f, indent=2)

        tracker.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": auc,
                "val_pr_auc": pr_auc,
                "lr": entry["lr"],
                "epoch_secs": dt,
            },
            step=epoch,
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "preprocessor": pickle.dumps(preproc),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_auc": auc,
                    "val_pr_auc": pr_auc,
                    "cat_vocab_sizes": preproc.cat_vocab_sizes,
                    "n_numeric": preproc.n_numeric,
                    "n_aggregate": preproc.n_aggregate,
                },
                str(best_path),
            )
            if progress:
                print(f"[train]   ✓ new best AUC={auc:.4f}, saved -> {best_path}")

    model_uri = tracker.log_pyfunc(
        checkpoint_path=best_path,
        customer_features_path=aggregates_path,
        registered_model_name=config.registered_model_name,
    )
    if progress and model_uri:
        print(f"[train] logged pyfunc model -> {model_uri}")

    return {
        "best_auc": best_auc,
        "history": history,
        "best_path": str(best_path),
        "metrics_path": str(metrics_path),
        "model_uri": model_uri,
    }
