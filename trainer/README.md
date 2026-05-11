# trainer

Тренирует бинарный классификатор фрода (FraudMLP) на 71 task.md-признаке.
Поддерживает legacy (data_augmented/) и production (partitioned parquets)
режимы, опциональный MLFlow tracking + pyfunc-регистрация в Model Registry.

## Pipeline

```
events parquet (data_augmented/ или ~/fraud/events/dt=*/)
   ├─ aggregate ─► customer_features.parquet  (49 per-customer фичей)
   └─ extract  ─► labelled_events.parquet     (87K supervised events)
                          │
                          ▼
                   Preprocessor (vocab + z-score)
                          │
                          ▼
                      FraudMLP
              (cat embeddings + numerics + aggs + has_hist)
                          │
                          ▼
              trainer/checkpoints/best.pt + metrics.json
              + MLFlow run + pyfunc model в Registry
```

## Тренировка — с нуля, не fine-tuning

Каждый вызов `trainer.cli train` создаёт **свежий** `FraudMLP(...)` и
обучает с zero-init весов. Старый `best.pt` НЕ загружается. Если нужен
warm-start — реализуйте отдельно (`model.load_state_dict(...)` перед
циклом эпох в `train.py:_train_impl`).

## Usage — legacy режим (на data_augmented/)

```bash
cd /home/clever/Documents/AntiFraud/AntiFraudMLWeb

# Одной командой
python3 -m trainer.cli all --input data_augmented

# Или по шагам
python3 -m trainer.cli aggregate --input data_augmented    # ~10 мин
python3 -m trainer.cli extract --input data_augmented      # ~2 мин
python3 -m trainer.cli train --input data_augmented        # ~5-10 мин GPU
```

## Usage — production режим (партиции от backend)

```bash
# Incremental aggregate — только новые партиции, остальное из state'а
python3 -m trainer.cli aggregate \
  --events-glob "/home/clever/fraud/events/dt=*/*.parquet" \
  --incremental \
  --state-path /home/clever/fraud/state/customer_features.state.parquet

# Extract с label-lag окном (7..30 дней назад от label_window_end)
python3 -m trainer.cli extract \
  --events-glob "/home/clever/fraud/events/dt=*/*.parquet" \
  --labels-glob "/home/clever/fraud/labels/dt=*/*.parquet" \
  --label-window-end 2026-05-05

# Train с MLFlow + регистрацией в Registry
python3 -m trainer.cli train \
  --mlflow-uri file:///home/clever/fraud/mlruns \
  --mlflow-experiment fraud_mlp_web \
  --registered-model-name fraud_mlp_web \
  --data-hash $(some_hash_command)
```

## Все CLI флаги

### Общие (все subcommand'ы)

| Флаг | Дефолт | Назначение |
|---|---|---|
| `--input` | `data_augmented` | директория с parquet'ами (legacy mode) |
| `--labels` | `data/train_labels.parquet` | путь к одиночному файлу меток (legacy) |
| `--out` | `trainer/checkpoints` | куда писать `best.pt` + `metrics.json` |
| `--quiet` | | заглушить прогресс |

### aggregate

| Флаг | Назначение |
|---|---|
| `--events-glob` | glob input partitions (repeatable) |
| `--incremental` | загрузить state и догрузить новые партиции |
| `--state-path` | где persistить raw sums (обязательно при `--incremental`) |
| `--batch-rows` | размер row-group батча (дефолт 200K) |

### extract

| Флаг | Назначение |
|---|---|
| `--events-glob` | glob event partitions (repeatable) |
| `--labels-glob` | glob label partitions (repeatable) |
| `--label-window-end` | если в метках есть `label_dttm`, держать только `[end-30d, end-7d]` |

### train

| Флаг | Дефолт | Назначение |
|---|---|---|
| `--epochs` | 20 | |
| `--batch-size` | 4096 | |
| `--lr` | 1e-3 | |
| `--weight-decay` | 1e-4 | |
| `--agg-dropout` | 0.20 | вероятность обнулить агрегаты + has_history (cold-start) |
| `--val-frac` | 0.20 | stratified split |
| `--seed` | 42 | |
| `--num-workers` | 2 | DataLoader workers |
| `--device` | auto | `cuda` или `cpu` |
| `--mlflow-uri` | `None` | если задан — оборачивает в `mlflow.start_run()` |
| `--mlflow-experiment` | `fraud_mlp_web` | |
| `--mlflow-run-name` | auto | |
| `--registered-model-name` | `None` | если задан — `mlflow.pyfunc.log_model(..., registered_model_name=...)` |
| `--data-hash` | `None` | свободный тег run'а (например DVC hash или git sha) |

## Структура

| Модуль | Назначение |
|---|---|
| `aggregate.py` | потоковая агрегация + persistent state для incremental режима |
| `extract.py` | join событий с метками (поддержка glob'ов и label-lag) |
| `preprocess.py` | picklable `Preprocessor` (vocab + z-score) — без sklearn |
| `dataset.py` | `FraudDataset` с per-sample aggregate dropout |
| `model.py` | `FraudMLP` — cat embeddings + 256→128→64→1, ReLU + BN + Dropout |
| `train.py` | training loop, AUC/PR-AUC, чекпоинт, опциональный MLFlow |
| `pyfunc.py` | `FraudPyfunc` — MLFlow PyFunc wrapper для inference-side load |
| `cli.py` | argparse dispatcher с поддержкой legacy + production режимов |

## Cold-start

20% обучающих сэмплов получают обнулённые агрегаты + `has_history=0`.
Модель учится падать обратно на event-level признаки для новых клиентов.
На инференсе клиент без записи в `customer_features.parquet` обрабатывается
тем же путём.

## MLFlow интеграция

При `--mlflow-uri ...`:

1. Run открывается с именем `fraud_mlp_e{epochs}_d{agg_dropout}_s{seed}` (override через `--mlflow-run-name`).
2. Логируются params (всё содержимое `TrainConfig`).
3. Tags: `git_sha`, `data_hash`, `device`.
4. На каждой эпохе: `train_loss`, `val_loss`, `val_auc`, `val_pr_auc`, `lr`, `epoch_secs`.
5. После цикла: `mlflow.pyfunc.log_model(...)` с артефактами
   `checkpoint` + `customer_features.parquet`, `code_paths=["trainer/"]`.
   Если задан `--registered-model-name`, новая версия регистрируется в Model Registry.

Без `--mlflow-uri` трекинг полностью no-op, `mlflow` в рантайме не
импортируется.

## Загрузка модели

**Production (через MLFlow Registry):**

```python
import mlflow.pyfunc, pandas as pd

model = mlflow.pyfunc.load_model("models:/fraud_mlp_web/Production")
df = pd.DataFrame([...])  # 71 task.md колонок
scores = model.predict(df)  # DataFrame с logit + p_fraud
```

**Dev (напрямую из best.pt):**

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

## Outputs

- `trainer/checkpoints/best.pt` — dict с `model_state`, pickled
  `preprocessor`, `config`, `epoch`, `val_auc`, `val_pr_auc`,
  `cat_vocab_sizes`, `n_numeric`, `n_aggregate`.
- `trainer/checkpoints/metrics.json` — history of per-epoch
  `{train_loss, val_loss, val_auc, val_pr_auc, lr, secs}`.
- MLFlow run (если включён) — params + metrics + tags + pyfunc model.
