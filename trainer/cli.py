"""CLI: aggregate / extract / train / all."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .aggregate import build_customer_features
from .extract import extract_labelled_events
from .train import TrainConfig, train


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", type=Path, default=Path("data_augmented"),
                   help="Directory with augmented parquet files.")
    p.add_argument("--labels", type=Path, default=Path("data/train_labels.parquet"),
                   help="Path to train_labels.parquet.")
    p.add_argument("--out", type=Path, default=Path("trainer/checkpoints"),
                   help="Output directory for model checkpoints + metrics.")
    p.add_argument("--quiet", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("aggregate", help="Build customer_features.parquet")
    _add_common(a)
    a.add_argument("--batch-rows", type=int, default=200_000)

    e = sub.add_parser("extract", help="Build labelled_events.parquet")
    _add_common(e)
    e.add_argument("--batch-rows", type=int, default=200_000)

    t = sub.add_parser("train", help="Train FraudMLP")
    _add_common(t)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=4096)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--agg-dropout", type=float, default=0.20)
    t.add_argument("--val-frac", type=float, default=0.20)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--num-workers", type=int, default=2)
    t.add_argument("--device", type=str, default=None,
                   help="cuda / cpu (default: auto).")

    all_ = sub.add_parser("all", help="Run aggregate -> extract -> train")
    _add_common(all_)
    all_.add_argument("--batch-rows", type=int, default=200_000)
    all_.add_argument("--epochs", type=int, default=20)
    all_.add_argument("--batch-size", type=int, default=4096)
    all_.add_argument("--lr", type=float, default=1e-3)
    all_.add_argument("--weight-decay", type=float, default=1e-4)
    all_.add_argument("--agg-dropout", type=float, default=0.20)
    all_.add_argument("--val-frac", type=float, default=0.20)
    all_.add_argument("--seed", type=int, default=42)
    all_.add_argument("--num-workers", type=int, default=2)
    all_.add_argument("--device", type=str, default=None)

    return p.parse_args(argv)


def _do_aggregate(args, progress: bool) -> None:
    build_customer_features(
        input_dir=args.input,
        output_path=args.input / "customer_features.parquet",
        batch_rows=args.batch_rows,
        progress=progress,
    )


def _do_extract(args, progress: bool) -> None:
    extract_labelled_events(
        input_dir=args.input,
        labels_path=args.labels,
        output_path=args.input / "labelled_events.parquet",
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
