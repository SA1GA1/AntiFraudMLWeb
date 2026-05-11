# trainer

Trains a binary fraud-detection classifier on the augmented hackathon dataset.

## Pipeline

```
data_augmented/*.parquet ──┬─► customer_features.parquet  (aggregate step)
                           └─► labelled_events.parquet    (extract step)
                                       │
                                       ▼
                              FraudMLP + Preprocessor
                                       │
                                       ▼
                          trainer/checkpoints/best.pt
                          trainer/checkpoints/metrics.json
```

## Usage

```bash
cd /home/clever/Documents/AntiFraud/AntiFraudMLWeb

# One-shot:
python3 -m trainer.cli all

# Or piecewise (useful for iteration):
python3 -m trainer.cli aggregate    # builds customer_features.parquet (~10 min)
python3 -m trainer.cli extract      # builds labelled_events.parquet  (~2 min)
python3 -m trainer.cli train        # trains, saves best.pt + metrics.json (~5-10 min on GPU)
```

Flags for `train`: `--epochs 20`, `--batch-size 4096`, `--lr 1e-3`,
`--agg-dropout 0.20`, `--val-frac 0.20`, `--seed 42`, `--device cuda`.

## Files

| Module | Purpose |
|---|---|
| `aggregate.py` | Streams pretrain+train+pretest, computes 49 per-customer features |
| `extract.py`   | Inner-joins train_part_* with train_labels → labelled_events.parquet |
| `preprocess.py`| Picklable `Preprocessor` (vocab + z-score) — no sklearn dep |
| `dataset.py`   | `FraudDataset` with per-sample aggregate dropout |
| `model.py`     | `FraudMLP` — categorical embeddings + 256→128→64→1 |
| `train.py`     | Training loop, AUC/PR-AUC, checkpoint, JSON metrics |
| `cli.py`       | argparse dispatcher (`aggregate / extract / train / all`) |

## Cold-start handling

During training, `--agg-dropout 0.20` randomly zeros each row's aggregate
vector (with `has_history=0`) on 20% of samples. The network learns to fall
back to event-level features when no customer history is available — useful
for unseen customers in production.

At inference time, any `customer_id` not present in `customer_features.parquet`
is handled the same way: aggregates → zeros, `has_history` → 0. The model
keeps producing scores.

## Outputs

- `trainer/checkpoints/best.pt` — dict with `model_state`, pickled
  `preprocessor`, `config`, `epoch`, `val_auc`, vocab sizes.
- `trainer/checkpoints/metrics.json` — list of per-epoch
  `{train_loss, val_loss, val_auc, val_pr_auc, lr, secs}`.

## Loading a saved model

```python
import torch, pickle
from trainer.model import FraudMLP

ck = torch.load("trainer/checkpoints/best.pt", weights_only=False)
preproc = pickle.loads(ck["preprocessor"])
model = FraudMLP(
    cat_vocab_sizes=ck["cat_vocab_sizes"],
    n_numeric=ck["n_numeric"],
    n_aggregate=ck["n_aggregate"],
)
model.load_state_dict(ck["model_state"])
model.eval()
```
