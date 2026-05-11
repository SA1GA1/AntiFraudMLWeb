# Данные

## Исходные временные периоды

Данные разделены на 4 периода:

- **Pre-train**, с 2023-10-01 по 2024-09-30 — только данные самих операций.
- **Train**, с 2024-10-01 по 2025-05-31 — операции + разметка целевой переменной.
- **Pre-test**, с 2025-06-01 по 2025-08-09 — начало тестовых данных с историей, но без разметки.
- **Test**, с 2025-06-01 по 2025-08-09 — финальный день тестовых клиентов, по которому требуется построить разметку.

Для удобства pre-train и train разбиты на 3 части. Каждый файл содержит свою группу клиентов на основе `customer_id`. Записи отсортированы по клиентам и времени операций.

## Целевая переменная

- `target = 1` — неподтверждённая операция (🔴, фрод)
- `target = 0` — напрямую подтверждённая клиентом операция (🟡)
- Все остальные операции в данных — без обратной связи (🟢, не используются для supervised обучения)

## Файлы в `data/`

| Файл | Период | Размер | Содержание |
|---|---|---:|---|
| `pretrain_part_1.parquet` | Pre-train | 625 MB | Первая треть клиентов |
| `pretrain_part_2.parquet` | Pre-train | 623 MB | Вторая треть клиентов |
| `pretrain_part_3.parquet` | Pre-train | 623 MB | Финальная треть клиентов |
| `train_part_1.parquet` | Train | 688 MB | Первая треть клиентов |
| `train_part_2.parquet` | Train | 687 MB | Вторая треть клиентов |
| `train_part_3.parquet` | Train | 688 MB | Финальная треть клиентов |
| `pretest.parquet` | Pre-test | 339 MB | История тестовых клиентов до последнего дня |
| `test.parquet` | Test | 17 MB | Финальный день, нужно предсказать target |
| `train_labels.parquet` | — | 1.2 MB | 87 514 пар `(customer_id, event_id) → target` |
| `sample_submit.csv` | — | 22 MB | Пример сабмишена: `event_id, predict` (raw score) |

## Объёмы

- **Pre-train:** ~91 M строк (3 файла)
- **Train:** ~85 M строк (3 файла)
- **Pre-test:** ~14 M строк
- **Test:** 633 683 строки
- **Размеченных событий:** 87 514 (из них 51 438 фрод / 36 076 норма, баланс 58.8% / 41.2%)
- **Уникальных клиентов в test:** 94 241

## Глоссарий исходных колонок (23)

| Колонка | Тип | Описание |
|---|---|---|
| `customer_id` | int64 | ID клиента банка |
| `event_id` | int64 | ID операции |
| `event_dttm` | str | Дата/время операции |
| `event_type_nm` | int32 | Тип операции (закодирован) |
| `event_desc` | int32 | Описание операции (закодировано) |
| `channel_indicator_type` | int32 | Канал совершения операции |
| `channel_indicator_sub_type` | int32 | Подтип канала |
| `operaton_amt` | double | Сумма операции в рублях |
| `currency_iso_cd` | int32 | Числовой ISO-код валюты |
| `mcc_code` | str | MCC merchant category code |
| `pos_cd` | int32 | Закодированный point-of-sale condition code |
| `accept_language` | str | Язык HTTP заголовка Accept-Language |
| `browser_language` | str | Язык браузера |
| `timezone` | int32 | Часовой пояс |
| `session_id` | int64 | Идентификатор сессии (см. ниже про NULL) |
| `operating_system_type` | int32 | Тип ОС (закодирован) |
| `battery` | str | Заряд устройства (0-1 или 0-100) |
| `device_system_version` | str | Версия ОС |
| `screen_size` | str | Разрешение экрана (`WxH`) |
| `developer_tools` | str/int | Флаг developer options на устройстве |
| `phone_voip_call_state` | int32 | Флаг VoIP-звонка во время операции |
| `web_rdp_connection` | int32 | Флаг удалённого управления над устройством |
| `compromised` | str/int | Наличие root-доступа на устройстве |

### ⚠ Особенности `session_id`

| Файл | Доля заполненных `session_id` |
|---|---:|
| `pretrain_part_*` | **0%** (везде NULL) |
| `train_part_*` | ~60% |
| `pretest.parquet` | ~64% |
| `test.parquet` | ~66% |

В `feature_generator` для строк с пустым `session_id` подставляется `event_id` как seed, иначе все per-session синтетические признаки коллапсируют в одну константу для всех 90 M строк pretrain.

---

# Расширенный набор: `data_augmented/`

После прогона `python3 -m feature_generator.cli` рядом с `data/` появляется директория `data_augmented/` с теми же 8 файлами событий, но с **36 колонками** (23 исходных + 13 синтетических). Файлы `train_labels.parquet` и `sample_submit.csv` не дублируются.

## 13 синтетических признаков

Все генерируются детерминированно (SplitMix64-хеш от int-ключа + salt + global seed). При наличии метки `target` распределения параметризованы class-conditional bias (risk=0.85 для фрода, 0.10 для нормы), иначе вычисляются из существующих риск-флагов (`compromised`, `developer_tools`, `web_rdp_connection`, `phone_voip_call_state`).

| Колонка | Тип | Ключ стабильности | Назначение |
|---|---|---|---|
| `attestation_status` | str (4 кат.) | customer_id | Результат аппаратной аттестации ОС: `passed`, `failed_root`, `emulator`, `modified_firmware`. Детектирует рут, эмуляторы, модифицированные прошивки. |
| `app_background_events` | int32 ≥0 | session_id | Сворачивания приложения во время заполнения формы — параллельная работа с инструкциями мошенника или фоновые скрипты. |
| `clipboard_paste_ratio_mobile` | float32 [0,1] | session_id | Доля вставок из буфера обмена при вводе — копирование реквизитов из фишинга или автозаполнение. |
| `entry_source` | str (4 кат.) | event_id | Способ запуска сессии: `manual`, `push`, `deeplink`, `sms_link`. Отсекает ботов и переходы по фрод-URL. |
| `connection_type` | str (5 кат.) | session_id | Тип подключения: `wifi_home`, `cellular`, `wifi_public`, `vpn`, `tor`. Аномалии: публичный Wi-Fi, VPN, частая смена SIM. |
| `touch_typing_rhythm` | float32 | session_id | Коэффициент вариации интервалов нажатий: человек ~0.3-0.6, бот <0.1. |
| `sim_country_mismatch` | int8 (0/1) | customer_id | Флаг расхождения страны SIM, IP и часового пояса — прокси-сети, фермы дропперов. |
| `network_rtt_avg` | float32 (ms) | session_id | Среднее время сетевого отклика до сервера банка — удалённое управление, медленные туннели. |
| `biometric_entry_used` | int8 (0/1) | customer_id + event_id | Успешный вход через FaceID/TouchID вместо пароля — подтверждает физическое присутствие владельца. |
| `storage_free_percent` | float32 [0,100] | customer_id + event_id | Уровень свободной памяти — старые/перегруженные устройства часто принадлежат дропперам и ботам. |
| `battery_charging_state` | str (4 кат.) | event_id | Статус зарядки: `discharging`, `charging`, `full`, `plugged_24_7`. Последнее — автоматизированные фрод-устройства. |
| `screen_orientation_changes` | int32 ≥0 | session_id | Частота смены ориентации экрана — аномально высокая при RDP/VNC. |
| `debugger_attached` | int8 (0/1) | event_id | Подключение системного отладчика или инъекции (Frida/Xposed) — обход защит, взлом, анализ трафика. |

## Производные артефакты в `data_augmented/`

Дополнительно после прогона `python3 -m trainer.cli all`:

### `customer_features.parquet` — 100 000 строк × 24 колонок (9.7 MB)

Per-customer агрегаты по объединению `pretrain + train + pretest` (108 M строк), потоково. Используются как контекст истории клиента при инференсе.

| Группа | Колонки |
|---|---|
| **Идентификатор** | `customer_id` |
| **Счётчики** | `event_count` |
| **Суммы операций** | `amt_mean`, `amt_std`, `amt_max`, `amt_log_mean` |
| **Существующие риск-флаги (доли)** | `compromised_share`, `web_rdp_share`, `developer_tools_share`, `phone_voip_share` |
| **Синтетические числовые (средние)** | `mean_app_background_events`, `mean_clipboard_paste`, `mean_typing_rhythm`, `mean_rtt`, `mean_storage_free`, `mean_screen_orientation` |
| **Синтетические бинарные (доли)** | `biometric_share`, `debugger_share`, `sim_mismatch_share` |
| **Производные доли** | `attestation_failed_share`, `vpn_tor_share` |
| **Временные** | `hours_span` (max−min event_dttm в часах), `night_ops_share` (часы 0-5), `weekend_share` |

### `labelled_events.parquet` — 87 514 строк × 37 колонок (5.6 MB)

Inner-join `data_augmented/train_part_*.parquet` с `data/train_labels.parquet` по `(customer_id, event_id)`. Содержит все 36 колонок augmented-набора + колонку `target ∈ {0, 1}`. Используется как готовый supervised тренировочный набор.

## Формат submission

`submission.csv` со столбцами `event_id, predict`, где `predict` — непрерывный скор (raw logit или вероятность). Пример из `sample_submit.csv`:

```
event_id,predict
125854726334416,-0.984226069913737
125949211749418,-1.14122043189846
124437385134670,-0.869486989105666
...
```

## Pipeline для воспроизведения

```bash
# 1. Добавить 13 синтетических признаков в data_augmented/
python3 -m feature_generator.cli

# 2. Построить агрегаты и labelled-набор + обучить модель
python3 -m trainer.cli all

# 3. Инференс на test.parquet → submission.csv
python3 predict_example.py
```
