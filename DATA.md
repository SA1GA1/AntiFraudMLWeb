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

После прогона `python3 -m feature_generator.cli` рядом с `data/` появляется директория `data_augmented/` с теми же 8 файлами событий, но в **71-колоночной схеме `task.md`** (browser identity, mouse/keyboard биометрия, network, fingerprints, login/trust). Исходные 23 колонки в augmented-файлы не сохраняются — это полная замена. Файлы `train_labels.parquet` и `sample_submit.csv` не дублируются.

## 71 синтетический признак

Все генерируются детерминированно (SplitMix64-хеш от int-ключа + salt + global seed). При наличии метки `target` распределения параметризованы class-conditional bias (risk=0.85 для фрода, 0.10 для нормы), иначе вычисляются из существующих риск-флагов (`compromised`, `developer_tools`, `web_rdp_connection`, `phone_voip_call_state`). Полный перечень — в `task.md` и `feature_generator/README.md`.

Сводка по группам:

| Группа | Кол-во | Примеры |
|---|---:|---|
| Identity / passthrough | 10 | `customer_id`, `event_id`, `session_id`, `event_dttm`, `operaton_amt`, `currency_iso_cd`, `mcc_code`, `pos_cd`, `browser_language`, `accept_language` |
| Browser identity (per-customer стабильно) | 12 | `browser_fingerprint`, `user_agent`, `browser_name/version`, `os_type/version`, `screen_resolution`, `screen_color_depth`, `system_language`, `webgl_vendor`, `canvas_fingerprint`, `audio_fingerprint` |
| Privacy / security флаги | 6 | `is_developer_tools`, `is_headless_browser`, `is_incognito`, `is_vpn_detected`, `is_proxy_detected`, `is_tor_detected` |
| Network (per-session) | 5 | `ip_address_hash`, `connection_type`, `network_rtt_avg_ms`, `asn`, `isp_name` |
| Mouse биометрика | 4 | `mouse_velocity_avg`, `mouse_acceleration_avg`, `mouse_jitter_score`, `mouse_linearity_score` |
| Click / scroll | 4 | `click_duration_avg_ms`, `right_click_count`, `scroll_velocity_avg`, `double_click_count` |
| Keyboard биометрика | 3 | `keyboard_typing_speed_median_ms`, `keyboard_typing_speed_std_dev`, `keyboard_typing_rhythm_cv` |
| Form interactions | 13 | `backspace_ratio`, `clipboard_paste_ratio`, `copy_events_count`, `paste_events_count`, `tab_switch_count`, `focus_blur_count`, `form_fill_duration_sec`, `idle_time_before_submit_sec`, `error_correction_ratio`, `hover_time_avg_ms`, `drag_drop_events`, `resize_events_count`, `zoom_level` |
| Session shape | 2 | `session_duration_sec`, `pages_visited_count` |
| Login / trust | 6 | `login_method`, `failed_login_attempts`, `time_since_last_login_sec`, `is_new_device`, `is_new_browser`, `device_trust_score` |
| Temporal (из `event_dttm`) | 3 | `hour_of_day`, `day_of_week`, `timezone_offset` |
| Transaction enrichment | 2 | `merchant_name`, `transaction_type` |
| Misc | 1 | `installed_fonts_count` |

⚠ Часть source-полей (`os_type`, `os_version`, `screen_resolution`, `timezone_offset`, `browser_language`, `accept_language`) при наличии в исходниках берётся passthrough'ем, иначе генерируется детерминированно.

## Производные артефакты в `data_augmented/`

Дополнительно после прогона `python3 -m trainer.cli all` (или `aggregate` / `extract` по отдельности):

### `customer_features.parquet` — 100 000 строк × 50 колонок (customer_id + 49 агрегатов)

Per-customer агрегаты по объединению `pretrain + train + pretest` (108 M строк), потоково. Используются как контекст истории клиента при инференсе. Полный список — в `trainer/aggregate.py::FEATURE_COLUMNS`.

| Группа | Колонки |
|---|---|
| **Идентификатор** | `customer_id` |
| **Счётчики / суммы** | `event_count`, `amt_mean`, `amt_std`, `amt_max`, `amt_log_mean` |
| **Privacy флаги (доли)** | `dev_tools_share`, `headless_share`, `incognito_share`, `vpn_share`, `proxy_share`, `tor_share` |
| **Trust флаги (доли)** | `new_device_share`, `new_browser_share` |
| **Network** | `mean_rtt` |
| **Mouse биометрика (средние)** | `mean_mouse_velocity`, `mean_mouse_accel`, `mean_mouse_jitter`, `mean_mouse_linearity` |
| **Click / scroll (средние)** | `mean_click_duration`, `mean_right_clicks`, `mean_scroll_velocity`, `mean_double_click` |
| **Keyboard биометрика (средние)** | `mean_typing_median_ms`, `mean_typing_std`, `mean_typing_cv` |
| **Form interactions (средние)** | `mean_backspace`, `mean_clipboard_paste`, `mean_copy_events`, `mean_paste_events`, `mean_tab_switch`, `mean_focus_blur`, `mean_form_fill`, `mean_idle_before_submit`, `mean_error_correction`, `mean_hover_time`, `mean_drag_drop`, `mean_resize_events`, `mean_zoom_level` |
| **Session shape (средние)** | `mean_session_duration`, `mean_pages_visited` |
| **Прочее (средние)** | `mean_installed_fonts`, `mean_failed_logins`, `mean_time_since_login`, `mean_device_trust` |
| **Категориальные доли** | `foreign_isp_share`, `mobile_os_share` |
| **Временные** | `hours_span` (max−min event_dttm в часах), `night_ops_share` (часы 0-5), `weekend_share` |

### `labelled_events.parquet` — 87 514 строк × 72 колонки (71 task.md + `target`)

Inner-join `data_augmented/train_part_*.parquet` с `data/train_labels.parquet` по `(customer_id, event_id)`. Содержит все 71 колонок augmented-набора + колонку `target ∈ {0, 1}`. Используется как готовый supervised тренировочный набор.

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
# 1. Сгенерировать augmented-схему task.md (71 колонка) в data_augmented/
python3 -m feature_generator.cli

# 2. Построить агрегаты и labelled-набор + обучить модель
python3 -m trainer.cli all

# 3. Инференс на test.parquet → submission.csv
python3 predict_example.py
```
