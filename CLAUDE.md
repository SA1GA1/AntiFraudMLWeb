# ITParkHackathon — детекция фрода в банковских операциях

## Задача

Бинарная классификация: для каждого события (банковской транзакции) предсказать `target ∈ {0, 1}` — была ли операция неподтверждённой (фрод). Результат отправляется как CSV в формате `event_id, predict` (raw logit или другой непрерывный скор).

## Структура данных

Исходные файлы (10 шт.) в `data/`:

| Файл | Период | Размер | Что |
|---|---|---:|---|
| `pretrain_part_{1,2,3}.parquet` | 2023-10-01 — 2024-09-30 | ~625 MB × 3 | Только операции, без меток |
| `train_part_{1,2,3}.parquet` | 2024-10-01 — 2025-05-31 | ~688 MB × 3 | Операции + есть метки в отдельном файле |
| `pretest.parquet` | 2025-06-01 — 2025-08-09 | 339 MB | История тестовых клиентов перед последним днём |
| `test.parquet` | 2025-06-01 — 2025-08-09 | 17 MB | Финальный день тестовых клиентов — здесь нужно предсказать target |
| `train_labels.parquet` | — | 1.2 MB | 87 514 пар `(customer_id, event_id) → target` |
| `sample_submit.csv` | — | 22 MB | Формат submission'а: `event_id, predict` |

### Схема исходных событий (23 колонки)

`customer_id (int64), event_id (int64), event_dttm (str datetime), event_type_nm, event_desc, channel_indicator_type, channel_indicator_sub_type, operaton_amt, currency_iso_cd, mcc_code, pos_cd, accept_language, browser_language, timezone, session_id (int64, частично NULL), operating_system_type, battery (str), device_system_version, screen_size (str), developer_tools, phone_voip_call_state, web_rdp_connection, compromised`.

**Важно:** `session_id` заполнен неравномерно:
- pretrain_part_*: **0%** (везде NULL)
- train_part_*: ~60%
- pretest: ~64%
- test: ~66%

Это причина первого бага в `feature_generator` (см. ниже).

### Объёмы

- ~85 M строк в train (3 файла), ~91 M в pretrain (3 файла), ~14 M в pretest, 633 683 в test
- Среди 87 514 размеченных событий: **51 438 фрод (target=1)**, **36 076 норма (target=0)** — баланс 58.8% / 41.2%
- 94 241 уникальных клиента в test; 100 000 клиентов всего в pretrain+train+pretest

## Что построено

```
ITParkHackathon/
├── data/                          # Исходные parquet'ы (не трогаем)
├── data_augmented/                # Парные файлы + дополнительные артефакты
│   ├── *.parquet                  # 8 augmented файлов (23 + 13 = 36 колонок)
│   ├── customer_features.parquet  # 100 K клиентов × 23 агрегата
│   └── labelled_events.parquet    # 87 514 размеченных событий с фичами
├── feature_generator/             # Этап 1: добавление 13 синтетических признаков
├── trainer/                       # Этап 2: обучение MLP
│   ├── checkpoints/
│   │   ├── best.pt                # 313 KB — лучшая модель + препроцессор
│   │   └── metrics.json           # История метрик по эпохам
│   └── ...
├── predict_example.py             # Пример инференса — пишет submission.csv
├── submission.csv                 # Результат инференса по test.parquet
├── DATA.md                        # Описание данных (от организаторов)
└── CLAUDE.md                      # Этот файл
```

## Этап 1 — feature_generator

Генерирует 13 поведенческих/устройство-интегритетных признаков и записывает их рядом с исходными колонками в `data_augmented/`.

### 13 признаков

| Колонка | Тип | Ключ | Назначение |
|---|---|---|---|
| `attestation_status` | str | customer_id | passed/failed_root/emulator/modified_firmware |
| `app_background_events` | int32 | session_id | сворачивания во время заполнения формы |
| `clipboard_paste_ratio_mobile` | float32 [0,1] | session_id | доля вставок из буфера |
| `entry_source` | str | event_id | manual/push/deeplink/sms_link |
| `connection_type` | str | session_id | wifi_home/cellular/wifi_public/vpn/tor |
| `touch_typing_rhythm` | float32 | session_id | коэффициент вариации интервалов нажатий (бот → ~0) |
| `sim_country_mismatch` | int8 | customer_id | флаг расхождения страны SIM/IP/TZ |
| `network_rtt_avg` | float32 (ms) | session_id | средний сетевой RTT |
| `biometric_entry_used` | int8 | customer_id + event_id | FaceID/TouchID |
| `storage_free_percent` | float32 | customer_id + event_id | свободное место |
| `battery_charging_state` | str | event_id | discharging/charging/full/plugged_24_7 |
| `screen_orientation_changes` | int32 | session_id | смены ориентации (RDP/демонстрация) |
| `debugger_attached` | int8 | event_id | Frida/Xposed-инъекции |

### Дизайн генератора

- **Детерминированность:** все случайные значения — SplitMix64-хеш от `(int_key, salt, seed)`, где key — `customer_id`, `session_id` или `event_id` в зависимости от семантики признака. Перезапуск с тем же seed → побитово идентичный parquet.
- **Class-conditional bias:** для train-файлов левый join с `train_labels.parquet`. Если target известен — risk = 0.85 (фрод) или 0.10 (норма); если нет — risk вычисляется из существующих флагов `compromised`, `developer_tools`, `web_rdp_connection`, `phone_voip_call_state`.
- **Параметры распределений** линейно интерполируются между benign- и fraud-вариантами по risk.
- **Fallback для пустого session_id:** там, где `session_id` NULL (особенно pretrain — 100% null), вместо `0` подставляется `event_id`. Иначе все session-keyed признаки коллапсируют в одну константу для всего pretrain. **Это был баг первой версии — починен.**

### Запуск

```bash
python3 -m feature_generator.cli                  # все 8 файлов
python3 -m feature_generator.cli --files test.parquet
```

### Что важно знать про эти признаки

13 признаков сгенерированы **conditionально на target** (где он известен) и **на существующие риск-флаги** (везде). Это значит:
- Они **очень сильный сигнал** для модели — на размеченной выборке любая разумная архитектура даст AUC≈1.0.
- На реальных данных, где такие сигналы должны измеряться датчиками, корреляция будет на порядок слабее.
- Это **не баг, а цель**: пользователь явно попросил «class-conditional bias, чтобы признаки были полезны для обучения NN».

## Этап 2 — trainer

Обучает бинарный классификатор (PyTorch MLP) на 87 514 размеченных событиях с обогащением 23 агрегатами по истории клиента.

### Пайплайн

```
data_augmented/*.parquet
   ├── aggregate ───► customer_features.parquet (100 K × 23 фичи)
   └── extract  ───► labelled_events.parquet  (87 K × 36 + target)
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

- 16 категориальных колонок → эмбеддинги размера `min(32, ⌈√vocab⌉)`. Cardinality cap = 1024 с бакетом OTHER.
- 16 числовых → z-score (mean/std из train-сплита, NaN → 0).
- 23 агрегата → z-score.
- `has_history` (0/1) — флаг наличия агрегатов для клиента.
- Конкатенация → Linear(IN, 256) + BN + ReLU + Dropout(0.3) → Linear(256, 128) + BN + ReLU + Dropout(0.3) → Linear(128, 64) + ReLU → Linear(64, 1).
- Loss: `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`.
- Optimizer: AdamW(lr=1e-3, wd=1e-4), CosineAnnealingLR.

### Cold-start через aggregate dropout

В 20% обучающих сэмплов агрегаты обнуляются и `has_history=0`. Это учит модель работать без истории клиента — нужно для production на новых клиентах, которых нет в `customer_features.parquet`. При инференсе клиенты без агрегатов автоматически получают нули + `has_history=0`.

### 23 агрегата по клиенту

`event_count`, `amt_mean/std/max/log_mean`, `compromised_share`, `web_rdp_share`, `developer_tools_share`, `phone_voip_share`, `mean_app_background_events`, `mean_clipboard_paste`, `mean_typing_rhythm`, `mean_rtt`, `mean_storage_free`, `mean_screen_orientation`, `biometric_share`, `debugger_share`, `sim_mismatch_share`, `attestation_failed_share`, `vpn_tor_share`, `hours_span`, `night_ops_share`, `weekend_share`.

Считаются по объединению pretrain + train + pretest (108 M строк), потоково в батчах 200 K строк. Накопитель — pandas DataFrame, `add(..., fill_value=0)` после каждого groupby.

### Известная утечка

Агрегаты по клиенту считаются по всему историческому набору, **включая** размеченные события. Метки сами не утекают (label-only поле не агрегируется), но синтетические признаки уже зависели от target, поэтому `attestation_failed_share`, `debugger_share`, и т.д. сильно коррелируют с долей фрода клиента. Это вместе с conditional-генерацией событийных признаков даёт AUC=1.0 — **по плану**, для baseline это приемлемо.

### Запуск

```bash
python3 -m trainer.cli all          # aggregate → extract → train (~12 минут)
python3 -m trainer.cli aggregate    # только агрегация (~10 минут)
python3 -m trainer.cli extract      # только извлечение размеченных (~2 минуты)
python3 -m trainer.cli train --epochs 20 --batch-size 4096   # только обучение (~45 с на RTX 4060)
```

### Текущий результат

20 эпох на RTX 4060 Laptop = 43.6 с. val AUC = 1.0 со 1-й эпохи (см. «известная утечка»), val_loss падает до 0.00017.

## Этап 3 — predict_example.py

Демонстрация использования модели:
1. Загружает `best.pt`, восстанавливает `Preprocessor` из pickle.
2. Batch-инференс по `test.parquet` → `submission.csv` (633 683 строки).
3. Одиночное предсказание по словарю-событию (для иллюстрации API).

```bash
python3 predict_example.py
```

## Окружение

- Python 3.14, pandas 3.0.1, pyarrow 23.0.1, numpy 2.4.4, torch 2.9.1+cu130
- GPU: NVIDIA RTX 4060 Laptop, 8.2 GB VRAM, CUDA available
- Project root: `/home/clever/Documents/ITParkHackathon`
- Никаких внешних БД нет — всё на parquet'ах в файловой системе

## Подводные камни (gotchas)

1. **pandas 3.0 datetime resolution = microseconds** по умолчанию, не nanoseconds. В `aggregate.py:_prepare_batch` конвертация в секунды через `dt.astype("datetime64[s]").view("int64")` — устойчиво к обеим резолюциям.
2. **`Series.view` удалён в pandas 3.0** — использовать `to_numpy().view(...)` или `astype(...)`.
3. **`session_id` в pretrain 100% NULL** — fallback к `event_id` обязателен, иначе все session-keyed фичи коллапсируют.
4. **Тип чекпоинта при загрузке:** `torch.load(..., weights_only=False)` нужен, потому что в чекпоинте лежит pickled-препроцессор (произвольный объект).
5. **AUC=1.0 не баг.** Это следствие того, что синтетические фичи генерируются conditional на target. Чтобы проверить event-level сигнал отдельно, запустить `train --agg-dropout 1.0` (всё равно AUC=1.0).

## Дальнейшие улучшения (не реализованы)

- **Без утечки агрегатов:** считать аггрегаты по клиенту, исключая размеченные события (или используя time-based split).
- **Сабмишен через CLI:** добавить `python -m trainer.cli predict` вместо отдельного скрипта.
- **Sequence-модель:** GRU/Transformer по истории клиента вместо плоского MLP.
- **Time-based валидация:** холдаут по дате, а не stratified random.
- **Semi-supervised pretrain:** masked-feature reconstruction на pretrain без меток.

## Полезные команды

```bash
# Регенерация augmented файлов
python3 -m feature_generator.cli

# Полный цикл обучения с нуля
python3 -m trainer.cli all

# Только инференс на test
python3 predict_example.py

# Проверка чекпоинта
python3 -c "import torch; ck = torch.load('trainer/checkpoints/best.pt', weights_only=False); print('AUC:', ck['val_auc'], 'epoch:', ck['epoch'])"

# Просмотр истории метрик
python3 -c "import json; print(json.dumps(json.load(open('trainer/checkpoints/metrics.json'))['history'][-1], indent=2))"
```
