"""CLI: aggregate / extract / train / all.

Supports two operating modes:

- **legacy** — directory + filename list (defaults match historical baseline,
  reads ``data_augmented/train_part_*.parquet`` and friends).
- **production** — explicit globs over partitioned parquet trees produced by
  ``AntiFraudMain`` (``--events-glob "/var/fraud/events/dt=2026-05-*/part-*.parquet"``).

MLFlow tracking is opt-in via ``--mlflow-uri``; if absent, training runs without
mlflow as a dependency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .aggregate import build_customer_features
from .extract import extract_labelled_events
from .train import TrainConfig, train


# ---------------------------------------------------------------------------
# Flag groups
# ---------------------------------------------------------------------------

def _add_io(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", type=Path, default=Path("data_augmented"),
                   help="Directory with parquet event files (legacy mode).")
    p.add_argument("--labels", type=Path, default=Path("data/train_labels.parquet"),
                   help="Path to single labels parquet (legacy mode).")
    p.add_argument("--out", type=Path, default=Path("trainer/checkpoints"),
                   help="Output directory for model checkpoints + metrics.")
    p.add_argument("--quiet", action="store_true")


def _add_events_glob(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--events-glob",
        action="append",
        default=None,
        help=("Glob pattern matching event parquets. Repeatable. Overrides "
              "--input/sources. Example: --events-glob "
              "'/var/fraud/events/dt=*/part-*.parquet'."),
    )


def _add_labels_glob(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--labels-glob",
        action="append",
        default=None,
        help=("Glob pattern matching label parquets. Repeatable. Overrides "
              "--labels. Example: --labels-glob '/var/fraud/labels/dt=*/part-*.parquet'."),
    )
    p.add_argument(
        "--label-window-end",
        type=str,
        default=None,
        help=("ISO date — if labels contain `label_dttm`, keep only rows in "
              "[end-30d, end-7d]. Used to respect chargeback lag."),
    )


def _add_aggregate_mode(p: argparse.ArgumentParser) -> None:
    p.add_argument("--batch-rows", type=int, default=200_000)
    p.add_argument(
        "--incremental",
        action="store_true",
        help=("Reuse persisted raw sums from --state-path and only process "
              "new partitions. Pass exclusively new files via --events-glob."),
    )
    p.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Where to persist aggregator state (raw sums + extremes).",
    )


def _add_train_hparams(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--agg-dropout", type=float, default=0.20)
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu (default: auto).")


def _add_mlflow(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help=("MLFlow tracking URI (e.g. file:///var/fraud/mlruns or "
              "http://localhost:5000). If unset, training runs without "
              "mlflow tracking."),
    )
    p.add_argument("--mlflow-experiment", type=str, default="fraud_mlp_web")
    p.add_argument("--mlflow-run-name", type=str, default=None)
    p.add_argument(
        "--registered-model-name",
        type=str,
        default=None,
        help=("If set, registers the trained pyfunc in the MLFlow Model "
              "Registry under this name. Typical value: 'fraud_mlp_web'."),
    )
    p.add_argument(
        "--data-hash",
        type=str,
        default=None,
        help=("Free-form hash of input data (e.g. DVC content hash). "
              "Logged as a tag on the MLFlow run."),
    )


# ---------------------------------------------------------------------------
# Subparser wiring
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("aggregate", help="Build customer_features.parquet")
    _add_io(a); _add_events_glob(a); _add_aggregate_mode(a)

    e = sub.add_parser("extract", help="Build labelled_events.parquet")
    _add_io(e); _add_events_glob(e); _add_labels_glob(e)
    e.add_argument("--batch-rows", type=int, default=200_000)

    t = sub.add_parser("train", help="Train FraudMLP")
    _add_io(t); _add_train_hparams(t); _add_mlflow(t)

    all_ = sub.add_parser("all", help="Run aggregate -> extract -> train")
    _add_io(all_)
    _add_events_glob(all_); _add_labels_glob(all_); _add_aggregate_mode(all_)
    _add_train_hparams(all_); _add_mlflow(all_)

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _do_aggregate(args, progress: bool) -> None:
    output = (args.input / "customer_features.parquet"
              if args.input else Path("customer_features.parquet"))
    build_customer_features(
        input_dir=args.input,
        output_path=output,
        events_glob=args.events_glob,
        incremental=getattr(args, "incremental", False),
        state_path=getattr(args, "state_path", None),
        batch_rows=args.batch_rows,
        progress=progress,
    )


def _do_extract(args, progress: bool) -> None:
    output = (args.input / "labelled_events.parquet"
              if args.input else Path("labelled_events.parquet"))
    extract_labelled_events(
        input_dir=args.input,
        labels_path=args.labels,
        output_path=output,
        events_glob=args.events_glob,
        labels_glob=args.labels_glob,
        label_window_end=getattr(args, "label_window_end", None),
        batch_rows=args.batch_rows,
        progress=progress,
    )


def _do_train(args, progress: bool) -> None:
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        agg_dropout=args.agg_dropout,
        val_frac=args.val_frac,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device or TrainConfig().device,
        mlflow_uri=getattr(args, "mlflow_uri", None),
        mlflow_experiment=getattr(args, "mlflow_experiment", "fraud_mlp_web"),
        mlflow_run_name=getattr(args, "mlflow_run_name", None),
        registered_model_name=getattr(args, "registered_model_name", None),
        data_hash=getattr(args, "data_hash", None),
    )
    train(
        labelled_path=args.input / "labelled_events.parquet",
        aggregates_path=args.input / "customer_features.parquet",
        output_dir=args.out,
        config=cfg,
        progress=progress,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    progress = not args.quiet

    if args.cmd == "aggregate":
        _do_aggregate(args, progress)
    elif args.cmd == "extract":
        _do_extract(args, progress)
    elif args.cmd == "train":
        _do_train(args, progress)
    elif args.cmd == "all":
        _do_aggregate(args, progress)
        _do_extract(args, progress)
        _do_train(args, progress)
    else:
        print(f"unknown command: {args.cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
