# AntiFraudMLWeb — детекция фрода в веб-сессиях банка

## Задача

Бинарная классификация: для каждого события (банковской операции) предсказать
`target ∈ {0, 1}` — фрод или нет. Submission — CSV в формате
`event_id, predict` (raw logit или произвольный непрерывный скор).

Целевая схема — **71 столбец** из `task.md`: web-fraud (browser identity,
mouse / keyboard биометрия, network, fingerprints, login/trust).

## Структура данных

Исходные файлы (10 шт.) в `data/`:

| Файл | Период | Размер | Что |
|---|---|---:|---|
| `pretrain_part_{1,2,3}.parquet` | 2023-10-01 — 2024-09-30 | ~625 MB × 3 | Только операции, без меток |
| `train_part_{1,2,3}.parquet` | 2024-10-01 — 2025-05-31 | ~688 MB × 3 | Операции + метки в отдельном файле |
| `pretest.parquet` | 2025-06-01 — 2025-08-09 | 339 MB | История тестовых клиентов перед последним днём |
| `test.parquet` | 2025-06-01 — 2025-08-09 | 17 MB | Финальный день тестовых клиентов |
| `train_labels.parquet` | — | 1.2 MB | 87 514 пар `(customer_id, event_id) → target` |
| `sample_submit.csv` | — | 22 MB | Формат submission'а: `event_id, predict` |

### Схема исходных событий (23 колонки в `data/`)

`customer_id, event_id, event_dttm, event_type_nm, event_desc,
channel_indicator_type, channel_indicator_sub_type, operaton_amt,
currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language,
timezone, session_id (часто NULL в pretrain), operating_system_type,
battery, device_system_version, screen_size, developer_tools,
phone_voip_call_state, web_rdp_connection, compromised`.

**Важно:** `session_id` заполнен неравномерно:
- pretrain_part_*: **0%** (везде NULL)
- train_part_*: ~60%
- pretest: ~64%
- test: ~66%

Поэтому в `feature_generator` есть fallback: при NULL `session_id` подставляется
`event_id` (иначе все session-keyed фичи коллапсируют в одну константу).

### Объёмы

- ~85 M строк в train (3 файла), ~91 M в pretrain (3 файла), ~14 M в pretest, 633 683 в test
- Среди 87 514 размеченных событий: **51 438 фрод (target=1)**, **36 076 норма (target=0)** — баланс 58.8% / 41.2%
- 94 241 уникальных клиента в test; 100 000 клиентов всего в pretrain+train+pretest

## Что построено

```
AntiFraudMLWeb/
├── data/                          # Исходные parquet'ы (не трогаем)
├── data_augmented/                # 8 augmented файлов (71 task.md колонка)
│   ├── *.parquet
│   ├── customer_features.parquet  # 100 K клиентов × 49 агрегатов
│   ├── customer_features.state.parquet  # raw sums для incremental aggregate
│   └── labelled_events.parquet    # 87 514 размеченных событий с фичами
├── feature_generator/             # Этап 1 (DEV/CI only): синтетические признаки
├── trainer/                       # Этап 2: обучение MLP
│   ├── checkpoints/
│   │   ├── best.pt                # чекпоинт + pickled preprocessor
│   │   └── metrics.json           # история метрик
│   ├── pyfunc.py                  # MLFlow PyFunc wrapper для inference
│   └── ...
├── orchestration/                 # Этап 4: Prefect daily flow + промоушен
│   ├── daily_flow.py
│   ├── deployment.py              # cron-расписание
│   ├── promote.py                 # AUC-gate перед Production
│   ├── notify.py                  # POST /admin/reload-model в AntiFraudMain
│   └── config.py                  # env-driven параметры
├── predict_example.py             # [DEPRECATED] локальный inference c best.pt
├── predict_from_json.py           # [DEPRECATED] JSON-инференс
├── predict_input_example.json     # Демо payload (web-fraud схема)
├── test1.json / test2_new_customer.json / test3_collision.json  # тесты
├── submission.csv                 # Результат инференса по test.parquet
├── task.md                        # ЦЕЛЕВАЯ схема (71 колонка)
├── DATA.md                        # Описание исходных данных
├── update.md                      # MLOps-обсуждение (MLFlow / Prefect)
└── CLAUDE.md                      # Этот файл
```

## Этап 1 — feature_generator (DEV / CI only)

**В production-пайплайне не вызывается.** Используется для:
- генерации seed-датасета новым разработчикам, у которых нет доступа к
  реальным данным;
- smoke-тестов препроцессора и trainer'а в CI;
- воспроизведения исходного синтетического baseline'а.

В ежедневном retrain'е (`orchestration/daily_flow.py`) на его место становится
parquet-партиции от `AntiFraudMain` (`/var/fraud/events/dt=*/...`).

Генерирует web-fraud признаки и записывает **только 71 колонку task.md**
(старые исходные колонки в augmented parquet не сохраняются).

### 71 колонка task.md (по группам)

**Identity / passthrough из source:** `customer_id`, `event_id`,
`session_id` (с fallback на `event_id` при NULL), `event_dttm`,
`operaton_amt`, `currency_iso_cd`, `mcc_code`, `pos_cd`,
`browser_language`, `accept_language`.

**Browser identity** (per-customer стабильно): `browser_fingerprint`,
`user_agent`, `browser_name`, `browser_version`, `os_type`,
`os_version`, `screen_resolution`, `screen_color_depth`,
`system_language`, `webgl_vendor`, `canvas_fingerprint`,
`audio_fingerprint`.

**Privacy / security флаги** (event/session, risk-biased):
`is_developer_tools`, `is_headless_browser`, `is_incognito`,
`is_vpn_detected`, `is_proxy_detected`, `is_tor_detected`.

**Network** (per-session): `ip_address_hash`, `connection_type`,
`network_rtt_avg_ms`, `asn`, `isp_name`.

**Mouse биометрика** (per-session, бот-bias под высокий risk):
`mouse_velocity_avg`, `mouse_acceleration_avg`, `mouse_jitter_score`,
`mouse_linearity_score`.

**Click / scroll:** `click_duration_avg_ms`, `right_click_count`,
`scroll_velocity_avg`, `double_click_count`.

**Keyboard биометрика:** `keyboard_typing_speed_median_ms`,
`keyboard_typing_speed_std_dev`, `keyboard_typing_rhythm_cv`.

**Form interactions** (per-session): `backspace_ratio`,
`clipboard_paste_ratio`, `copy_events_count`, `paste_events_count`,
`tab_switch_count`, `focus_blur_count`, `form_fill_duration_sec`,
`idle_time_before_submit_sec`, `error_correction_ratio`,
`hover_time_avg_ms`, `drag_drop_events`, `resize_events_count`,
`zoom_level`.

**Session shape:** `session_duration_sec`, `pages_visited_count`.

**Login / trust** (per-customer + per-event): `login_method`,
`failed_login_attempts`, `time_since_last_login_sec`, `is_new_device`,
`is_new_browser`, `device_trust_score`.

**Temporal** (из `event_dttm`): `hour_of_day`, `day_of_week`,
`timezone_offset`.

**Transaction enrichment:** `merchant_name`, `transaction_type`.

### Дизайн генератора

- **Детерминированность:** SplitMix64-хеш от `(int_key, salt, seed)`. Ключ —
  `customer_id`, `session_id` или `event_id` в зависимости от семантики.
  Перезапуск с тем же seed → побитово идентичный parquet.
- **Class-conditional bias:** для train-файлов левый join с
  `train_labels.parquet`. Если target известен — risk = 0.85 (фрод) или 0.10
  (норма); иначе risk вычисляется из source-флагов (`compromised`,
  `developer_tools`, `web_rdp_connection`, `phone_voip_call_state`).
- **Параметры распределений** линейно интерполируются между benign- и
  fraud-вариантами по risk. Бинарные флаги — `Bernoulli(lerp(p_benign, p_fraud, r))`.
- **Source passthrough:** `os_type` / `os_version` / `screen_resolution` /
  `timezone_offset` / `browser_language` / `accept_language` берутся из
  исходных колонок если они есть, иначе генерируются.

### Запуск

```bash
python3 -m feature_generator.cli                  # все 8 файлов
python3 -m feature_generator.cli --files test.parquet
```

### Что важно знать про эти признаки

Признаки сгенерированы **conditionально на target** (где он известен) и
**на source risk-флаги** (везде). Это значит:
- Они **очень сильный сигнал** для модели — на размеченной выборке любая
  разумная архитектура даст AUC≈1.0.
- На реальных web-сессиях, где такие сигналы должны измеряться
  фронтовыми SDK / fingerprinting JS, корреляция будет слабее.
- Это **не баг, а цель**: для baseline нужны полезные для NN признаки.

## Этап 2 — trainer

Обучает бинарный классификатор (PyTorch MLP) на 87 514 размеченных событиях
с обогащением 49 агрегатами по истории клиента.

### Пайплайн

```
data_augmented/*.parquet
   ├── aggregate ───► customer_features.parquet (100 K × 49 фичей)
   └── extract  ───► labelled_events.parquet  (87 K × 71 + target)
                          │
                          ▼
                   Preprocessor (vocab + z-score)
                          │
                          ▼
                      FraudMLP
                  (categorical embeddings + numerics + aggs + has_history)
                          │
                          ▼
            best.pt  +  metrics.json
```

### Архитектура модели

- 19 категориальных колонок → эмбеддинги `min(32, ⌈√vocab⌉)`. Cardinality cap = 1024 с бакетом OTHER.
- 44 числовые → z-score (mean/std из train-сплита, NaN → 0).
- 49 агрегатов → z-score.
- `has_history` (0/1) — флаг наличия агрегатов для клиента.
- Конкатенация → Linear(IN, 256) + BN + ReLU + Dropout(0.3) → Linear(256, 128) + BN + ReLU + Dropout(0.3) → Linear(128, 64) + ReLU → Linear(64, 1).
- Loss: `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`.
- Optimizer: AdamW(lr=1e-3, wd=1e-4), CosineAnnealingLR.

### Что нумерик / что категориал

**Numeric (44):** `operaton_amt` (log1p), все `is_*` флаги, производный
`biometric_login` (из `login_method == 'biometric'`), все
mouse/click/scroll/keyboard/form-метрики, session-shape, login-trust,
`network_rtt_avg_ms`, `screen_color_depth`, `installed_fonts_count`,
`timezone_offset`, `asn`.

**Categorical (19):** `currency_iso_cd`, `mcc_code`, `pos_cd`,
`browser_name`, `browser_version`, `os_type`, `os_version`,
`screen_resolution`, `system_language`, `browser_language`,
`accept_language`, `merchant_name`, `transaction_type`,
`connection_type`, `isp_name`, `webgl_vendor`, `login_method`,
`hour_of_day`, `day_of_week`.

### Cold-start через aggregate dropout

В 20% обучающих сэмплов агрегаты обнуляются и `has_history=0`. Это учит модель
работать без истории клиента (для новых клиентов, которых нет в
`customer_features.parquet`). При инференсе клиенты без агрегатов получают
нули + `has_history=0`.

### 49 агрегатов по клиенту

`event_count`, `amt_mean/std/max/log_mean`, `dev_tools_share`, `headless_share`,
`incognito_share`, `vpn_share`, `proxy_share`, `tor_share`,
`new_device_share`, `new_browser_share`, `mean_rtt`,
`mean_mouse_velocity/accel/jitter/linearity`, `mean_click_duration`,
`mean_right_clicks`, `mean_scroll_velocity`, `mean_typing_median_ms/std/cv`,
`mean_backspace`, `mean_clipboard_paste`, `mean_copy_events/paste_events`,
`mean_tab_switch`, `mean_focus_blur`, `mean_form_fill`,
`mean_idle_before_submit`, `mean_error_correction`, `mean_hover_time`,
`mean_double_click`, `mean_drag_drop`, `mean_resize_events`,
`mean_zoom_level`, `mean_session_duration`, `mean_pages_visited`,
`mean_installed_fonts`, `mean_failed_logins`, `mean_time_since_login`,
`mean_device_trust`, `foreign_isp_share`, `mobile_os_share`, `hours_span`,
`night_ops_share`, `weekend_share`.

Считаются по объединению pretrain + train + pretest (108 M строк), потоково в
батчах 200 K строк. Накопитель — pandas DataFrame, `add(..., fill_value=0)`
после каждого groupby.

### Известная утечка

Агрегаты по клиенту считаются по всему историческому набору, **включая**
размеченные события. Метки не утекают напрямую (только их хеш-производные —
синтетические фичи зависят от target). Это даёт AUC=1.0 — **по плану**,
для baseline приемлемо.

### Запуск (legacy, на синтетике)

```bash
python3 -m trainer.cli all          # aggregate → extract → train (~12 минут)
python3 -m trainer.cli aggregate    # только агрегация (~10 минут)
python3 -m trainer.cli extract      # только извлечение размеченных (~2 минуты)
python3 -m trainer.cli train --epochs 20 --batch-size 4096   # только обучение
```

### Запуск (production, на партициях от backend)

```bash
# Инкрементальный пересчёт агрегатов по новым партициям
python3 -m trainer.cli aggregate \
  --events-glob "/var/fraud/events/dt=*/*.parquet" \
  --incremental \
  --state-path /var/fraud/state/customer_features.state.parquet

# Извлечение размеченных событий с label-lag окном
python3 -m trainer.cli extract \
  --events-glob "/var/fraud/events/dt=*/*.parquet" \
  --labels-glob "/var/fraud/labels/dt=*/*.parquet" \
  --label-window-end 2026-05-04

# Обучение с MLFlow-трекингом и регистрацией pyfunc
python3 -m trainer.cli train \
  --mlflow-uri file:///var/fraud/mlruns \
  --registered-model-name fraud_mlp_web \
  --data-hash $(date +%Y%m%d)-bootstrap
```

## Этап 3 — predict (DEPRECATED)

Production-инференс теперь живёт в `AntiFraudMain` и тянет модель из MLFlow
Registry (`models:/fraud_mlp_web/Production`). См. `update.md` раздел 3.2 и
`/home/clever/Documents/AntiFraud/AntiFraudMain/`.

`predict_example.py` / `predict_from_json.py` остаются в репо как
**dev-утилиты** для быстрой проверки чекпоинта сразу после тренировки.
Не использовать в production.

Готовые payload-файлы для проверки модели:

- `predict_input_example.json` — базовая пара fraud / benign.
- `test1.json` — известный клиент, контрольная пара.
- `test2_new_customer.json` — cold-start (нет в `customer_features.parquet`).
- `test3_collision.json` — «чистый» клиент с подозрительным событием
  (проверка, перетягивает ли история вердикт).

## Этап 4 — daily retrain (Prefect + MLFlow)

Ежедневный flow живёт в `orchestration/`. Запускается Prefect-агентом по
cron'у (по умолчанию 03:00 Europe/Moscow). Шаги:

1. `compute_data_hash` — отпечаток входных файлов (sha256 от path/size/mtime).
   Идёт как MLFlow run tag `data_hash`, привязывает модель к версии данных.
2. `run_aggregate` — `python -m trainer.cli aggregate --incremental ...`.
   Не пересобирает 108 M строк, дописывает только новые партиции.
   Идемпотентен по `(path, size, mtime)` — повторный запуск пропускает
   уже обработанные файлы.
3. `run_extract` — окно меток `[today-30d, today-7d]` (лаг чарджбэков).
   `--append` режим аккумулирует размеченные события между запусками.
4. `run_train` — обучение + MLFlow run + `mlflow.pyfunc.log_model(...)`
   с регистрацией в Registry под именем `fraud_mlp_web`. Пропускается
   если `labelled_events.parquet` пустой (нет новых меток).
5. `run_promote` — `validate_and_promote` в `orchestration/promote.py`:
   новая версия становится Production только если `val_auc ≥ prod_auc − 0.005`.
6. `run_notify` — POST `/admin/reload-model` в AntiFraudMain. Бэкенд
   подтягивает новую модель из Registry без рестарта.

Все пути и URL — env-driven (см. `orchestration/config.py`):
`FRAUD_ROOT`, `FRAUD_EVENTS_GLOB`, `FRAUD_WORK_DIR`, `MLFLOW_TRACKING_URI`,
`FRAUD_BACKEND_RELOAD_URL`, `FRAUD_CRON`, `FRAUD_CRON_TZ`, и т.д.

### Запуск daily flow

```bash
# Один раз — зарегистрировать deployment
python3 -m orchestration.deployment

# Запустить агента в фоне
prefect worker start --pool default &

# Принудительный прогон вручную (без cron'а)
python3 -m orchestration.daily_flow
```

### ⚠ Известные баги daily_flow

1. **Overwrites baseline в `data_augmented/`.** `daily_flow.run_aggregate` /
   `run_extract` вызывают `trainer.cli` без `--input`, поэтому используется
   дефолт `data_augmented`, и `customer_features.parquet` /
   `labelled_events.parquet` пишутся туда, перезаписывая исходный baseline
   (100 K customers, 87 K labelled events). До исправления — делать
   backup или менять дефолт в `_do_aggregate`/`_do_extract`.

2. **Падает на пустом `~/fraud/events/`.** `compute_data_hash` отрабатывает
   на пустом globe, но `run_extract` падает с "No labelled events found"
   если в events лежит только test.parquet (у него по определению нет
   меток). Решение для bootstrap'а — скопировать в events `train_part_*`
   с реальными метками.

3. **Тренировка с нуля, не fine-tuning.** `_train_impl` создаёт свежий
   `FraudMLP(...)` без `model.load_state_dict(previous_best.pt)`. Если
   нужен warm-start между runs — реализовать отдельно. Это означает,
   что `data_augmented/*.parquet` НЕ переиспользуется автоматически —
   trainer видит только то, что лежит в текущих outputs aggregate/extract.

## Окружение

- Python 3.14, pandas 2.3.3 (downgraded mlflow constraint), pyarrow 23.0.1,
  numpy 2.4.4, torch 2.9.1+cu130
- mlflow 3.12, prefect 3.7 — установлены в ~/.local
- GPU: NVIDIA RTX 4060 Laptop, 8.2 GB VRAM, CUDA available
- Project root: `/home/clever/Documents/AntiFraud/AntiFraudMLWeb`
- **Shared data root (dev):** `/home/clever/fraud/` — events, labels,
  state, mlruns, work. Override через `FRAUD_ROOT` env var.
- **Production canonical:** `/var/fraud/` (требует sudo на этой машине).
  Все примеры в коде используют `/var/fraud/` как иллюстрацию; реальные
  дефолты в `orchestration/config.py` указывают на `/home/clever/fraud/`.
- Никаких внешних БД нет — всё на parquet'ах в файловой системе.
- DVC экспериментально подключался, потом удалён — для single-host setup
  избыточен; reproducibility покрывает `compute_data_hash` + MLFlow tags.

## Подводные камни (gotchas)

1. **pandas 3.0 datetime resolution = microseconds** по умолчанию. В
   `aggregate.py:_prepare_batch` конвертация в секунды через
   `dt.astype("datetime64[s]").view("int64")` — устойчиво к обеим резолюциям.
2. **`Series.view` удалён в pandas 3.0** — использовать `to_numpy().view(...)`
   или `astype(...)`.
3. **`session_id` в pretrain 100% NULL** — fallback к `event_id`
   обязателен, иначе все session-keyed фичи коллапсируют.
4. **Тип чекпоинта при загрузке:** `torch.load(..., weights_only=False)`
   нужен, потому что в чекпоинте лежит pickled-препроцессор.
5. **AUC=1.0 не баг.** Это следствие того, что синтетические фичи генерируются
   conditional на target. Чтобы проверить event-level сигнал отдельно,
   запустить `train --agg-dropout 1.0` (всё равно AUC=1.0).
6. **Augmented parquet ≠ source parquet.** Augmented содержит **только** 71
   колонку task.md. Старые source-поля (`event_type_nm`, `event_desc`,
   `channel_indicator_type` и т.п.) в нём отсутствуют. Если нужны — читать
   из `data/`.

## Дальнейшие улучшения

### Сделано

- **MLFlow tracking + Model Registry** — `trainer/train.py` + `trainer/pyfunc.py`.
- **Prefect daily flow** — `orchestration/daily_flow.py`.
- **Validation gate перед Production** — `orchestration/promote.py`.
- **Hot-reload модели в бэкенде** — `orchestration/notify.py` →
  `AntiFraudMain /admin/reload-model`.
- **Инкрементальная агрегация** — `aggregate.save_state` / `load_state`.

### Открытые направления

- **Без утечки агрегатов:** считать аггрегаты по клиенту, исключая
  размеченные события (или используя time-based split).
- **Sequence-модель:** GRU/Transformer по истории клиента вместо плоского MLP.
- **Time-based валидация:** холдаут по дате, а не stratified random.
- **Semi-supervised pretrain:** masked-feature reconstruction на pretrain без меток.
- **Real fingerprinting integration:** заменить синтетические canvas/audio/webgl
  отпечатки данными из реального fingerprinting SDK (см. `AntiFraudMain/update.md`).
- **Evidently drift report** — добавить шаг в `daily_flow` после `extract`.
- **Pandera schema validation** — единый контракт между backend и trainer.

## Полезные команды

```bash
# Регенерация augmented файлов (dev/synthetic only)
python3 -m feature_generator.cli

# Полный цикл обучения с нуля (legacy)
python3 -m trainer.cli all

# Локальный inference c best.pt (dev only)
python3 predict_example.py
python3 predict_from_json.py test1.json

# Daily flow вручную
python3 -m orchestration.daily_flow

# DVC reproducible run (пересчёт только изменившихся стадий)
dvc repro

# MLFlow UI
mlflow ui --backend-store-uri file:///var/fraud/mlruns

# Проверка чекпоинта
python3 -c "import torch; ck = torch.load('trainer/checkpoints/best.pt', weights_only=False); print('AUC:', ck['val_auc'], 'epoch:', ck['epoch'])"

# Просмотр истории метрик
python3 -c "import json; print(json.dumps(json.load(open('trainer/checkpoints/metrics.json'))['history'][-1], indent=2))"

# Список версий в MLFlow Registry
python3 -c "from mlflow.tracking import MlflowClient; c = MlflowClient('file:///var/fraud/mlruns'); print([(v.version, v.current_stage) for v in c.search_model_versions(\"name='fraud_mlp_web'\")])"
```
