# AntiFraudMLWeb

ML-пайплайн детекции фрода в веб-сессиях банка. Бинарная классификация
по 71 task.md-признаку (browser identity, mouse/keyboard биометрика,
network, fingerprints, login/trust) — на каждое событие предсказывает
`P(fraud)`.

Часть триплета:

- **AntiFraudMLWeb** (этот репо) — тренировочный pipeline для web-фрода.
- **AntiFraudMLMobile** — параллельный для mobile-фрода (та же архитектура).
- **AntiFraudMain** — FastAPI inference-сервис, грузит модель из MLFlow
  Registry, скорит `/score/behavior`.

## Статус

**Bootstrap фаза.** Тренировочные данные — синтетические, сгенерированные
`feature_generator/` из реального но-обезличенного датасета. Backend
(`AntiFraudMain`) ещё не пишет события — реальный поток в `~/fraud/events/`
пуст. Daily flow готов, но запускается на legacy данных через копию
`data_augmented/*` → `~/fraud/events/dt=legacy/`.

См. `update.md` для MLOps-обсуждения и `AntiFraudMain/update.md` для
schema gap между backend (16 валидируемых полей) и trainer (71).

## Архитектура

```
┌─ feature_generator/ ──┐   ┌─ trainer/ ──────────┐   ┌─ orchestration/ ─┐
│ синтетика 71 колонки  │──►│ aggregate           │   │ daily_flow (cron)│
│ из data/ → data_augm/ │   │ extract             │   │   ↓              │
│ (dev/CI only)         │   │ train (FraudMLP)    │◄──┤  subprocess CLI  │
└───────────────────────┘   │ pyfunc (MLFlow log) │   │   ↓              │
                            └────────┬────────────┘   │ promote (AUC gate│
                                     │                │   ↓              │
                                     ▼                │ notify backend   │
                            MLFlow Registry           └──────────────────┘
                            (file:///~/fraud/mlruns)         │
                                     │                       │
                                     └────►  AntiFraudMain ◄─┘
                                            /admin/reload-model
```

## Quick start

### Уровень 1: тренировка на готовых артефактах (~3 минуты)

`data_augmented/customer_features.parquet` и `labelled_events.parquet` уже
лежат в репо (созданы предыдущими прогонами `feature_generator` +
`trainer.cli aggregate/extract`). Дообучение **с нуля** (не fine-tuning) —
PyTorch инициализируется свежим, не подхватывает старый `best.pt`.

```bash
python3 -m trainer.cli train \
  --input data_augmented \
  --mlflow-uri file:///home/clever/fraud/mlruns \
  --mlflow-experiment fraud_mlp_web \
  --registered-model-name fraud_mlp_web \
  --epochs 3 \
  --data-hash bootstrap-$(date +%Y%m%d)
```

После — `mlflow ui --backend-store-uri file:///home/clever/fraud/mlruns`
покажет run на http://localhost:5000.

### Уровень 2: полный legacy цикл (~12 минут)

Пересчитать агрегаты по 108 M строк, заново извлечь размеченные события,
обучить:

```bash
python3 -m trainer.cli all \
  --input data_augmented \
  --mlflow-uri file:///home/clever/fraud/mlruns \
  --registered-model-name fraud_mlp_web \
  --epochs 20 \
  --data-hash baseline-$(date +%Y%m%d)
```

### Уровень 3: daily flow (Prefect)

Требует чтобы `~/fraud/events/` содержала **train-партиции с метками**
(не test.parquet — у него по определению нет target). Если хочешь
прогнать на legacy:

```bash
mkdir -p ~/fraud/events/dt=legacy ~/fraud/labels/dt=legacy
cp data_augmented/{pretrain_part_*,train_part_*,pretest}.parquet ~/fraud/events/dt=legacy/
cp data/train_labels.parquet ~/fraud/labels/dt=legacy/

python3 -m orchestration.daily_flow
```

**⚠ Известный баг:** subprocess-вызовы в `daily_flow.py` пишут вывод
`aggregate`/`extract` в `data_augmented/`, перезаписывая baseline
`customer_features.parquet` (100K клиентов) и `labelled_events.parquet`
(87K событий). До исправления — делать backup перед запуском daily_flow.

## Структура

```
AntiFraudMLWeb/
├── data/                   # сырые parquet'ы — train_labels, sample_submit
├── data_augmented/         # 8 augmented parquet'ов (71 task.md колонка)
│   ├── *.parquet           # часть из них перенесена в ~/fraud/events/dt=0000-00-00/
│   ├── customer_features.parquet     # 100 K customers × 49 агрегатов (output legacy aggregate)
│   └── labelled_events.parquet       # 87 514 размеченных × 72 колонки (output legacy extract)
├── feature_generator/      # dev/CI: синтез 71 колонки task.md
├── trainer/                # aggregate / extract / train / pyfunc
│   ├── checkpoints/        # best.pt + metrics.json
│   ├── pyfunc.py           # FraudPyfunc — обёртка для MLFlow log_model
│   └── ...
├── orchestration/          # Prefect daily flow + promote + notify
├── predict_example.py      # [DEPRECATED] dev/debug local inference
├── predict_from_json.py    # [DEPRECATED] JSON inference
├── task.md                 # ЦЕЛЕВАЯ схема (71 колонка) — data dictionary
├── DATA.md                 # детальное описание данных
├── CLAUDE.md               # карта проекта для контекстных ассистентов
├── update.md               # MLOps-обсуждение (MLFlow / Prefect)
└── README.md               # этот файл
```

## Окружение

- Python 3.14, PyTorch 2.9 + CUDA, pandas 2.3, pyarrow 23.0, numpy 2.4
- mlflow 3.12, prefect 3.7 — установлены в `~/.local`
- GPU: NVIDIA RTX 4060 Laptop, 8 GB VRAM
- Shared dev paths: `~/fraud/{events,labels,state,mlruns,work}`
- Production canonical (требует sudo): `/var/fraud/...`
- Reproducibility — через `compute_data_hash` (SHA256 от path+size+mtime
  входных файлов), логируется как MLFlow run tag. См. секцию «Что дальше»
  если интересует история DVC-эксперимента.

## Что дальше

Открытые тикеты (по приоритету):

1. **Backend event-sink** — `AntiFraudMain` должен начать писать события
   в `~/fraud/events/dt=YYYY-MM-DD/part-*.parquet`. См.
   `AntiFraudMain/update.md` (раздел Schema gap) + плановый раздел в
   `update.md` (Architecture).
2. **Источник лейблов** — наполнять `~/fraud/labels/dt=YYYY-MM-DD/part-*.parquet`
   из внешних каналов: chargeback-фид от процессинга (лаг 1-30 дней),
   выгрузка решений fraud-команды из CRM, customer complaints, ручные
   отметки операторов. Без этого daily_flow на bootstrap'е переобучается
   на статичных 87 514 парах из `data/train_labels.parquet` и
   AUC-gate каждый день отвергает идентичную модель — реального
   ML-сигнала нет до подключения хотя бы одного источника.
3. **`/admin/reload-model`** в AntiFraudMain — без него `daily_flow.notify`
   падает после промоушена.
4. **Fix daily_flow output paths** — вынести `customer_features.parquet`/
   `labelled_events.parquet` из `data_augmented/` в `~/fraud/work/`.
5. **Frontend SDK биометрики** — собирать mouse/keyboard/canvas, без них
   55 task.md полей всегда NaN.
6. **Time-based валидация и aggregate без утечки** — см. `CLAUDE.md`
   «Дальнейшие улучшения».

## Лицензия / контекст

Внутренний проект банковской фрод-детекции. Данные обезличены.
Производные модели (`fraud_mlp_web`) регистрируются в локальном MLFlow,
не подразумеваются для публичного релиза.
