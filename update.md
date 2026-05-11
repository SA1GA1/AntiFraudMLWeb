# Update — MLOps-обсуждение

Конспект последнего обсуждения по проекту AntiFraudMLWeb: интеграция MLFlow,
ежедневное переобучение, разделение сервисов.

---

## 1. Использование MLFlow в проекте

MLFlow закроет три пробела, которые сейчас есть в пайплайне: отсутствие
истории экспериментов, единственный безымянный чекпоинт и привязка инференса
к локальному файлу `best.pt`.

### 1.1 Tracking — заменяет `metrics.json`

Текущий `trainer/train.py` пишет в `checkpoints/metrics.json` плоский список
метрик. Это работает для одного запуска, но не отвечает на вопросы «как
`--agg-dropout 0.3` сравнить с 0.2» или «какая ветка кода дала AUC=0.98».

В цикле обучения:

```python
with mlflow.start_run(run_name=f"fraud_mlp_e{args.epochs}_d{args.agg_dropout}"):
    mlflow.log_params(vars(args))                # все CLI-флаги
    mlflow.log_params({                          # внутренние размеры
        "n_numeric": preproc.n_numeric,
        "n_aggregate": preproc.n_aggregate,
        "cat_vocab_total": sum(preproc.cat_vocab_sizes.values()),
    })
    mlflow.set_tag("git_sha", subprocess.check_output(["git","rev-parse","HEAD"]).strip())

    for epoch in range(args.epochs):
        ...
        mlflow.log_metrics({"train_loss": ..., "val_auc": ..., "val_pr_auc": ...}, step=epoch)
```

После этого `mlflow ui` (на `file:./mlruns`) даёт сортировку/фильтр по
val_auc, диффы параметров, графики loss-кривых поверх нескольких ранов.

### 1.2 Artifacts + Model Registry — заменяют `best.pt`

Сейчас `best.pt` — это pickle с весами + Preprocessor + конфигом.
Самодостаточно, но без версионирования: после второй тренировки старый
чекпоинт перезаписывается.

В MLFlow это упаковывается как **PythonModel** (custom flavor), чтобы
preprocessor поехал вместе с моделью:

```python
class FraudPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        ck = torch.load(context.artifacts["checkpoint"], weights_only=False)
        self.preproc = pickle.loads(ck["preprocessor"])
        self.model = FraudMLP(...).eval()
        self.model.load_state_dict(ck["model_state"])

    def predict(self, context, df):
        X = self.preproc.transform(df, customer_features=...)
        with torch.no_grad():
            return torch.sigmoid(self.model(**X)).numpy()

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=FraudPyfunc(),
    artifacts={"checkpoint": "trainer/checkpoints/best.pt",
               "customer_features": "data_augmented/customer_features.parquet"},
    registered_model_name="fraud_mlp",
)
```

`fraud_mlp` появляется в registry → версии 1, 2, 3… со стадиями
Staging/Production. `predict_example.py` упрощается до:

```python
model = mlflow.pyfunc.load_model("models:/fraud_mlp/Production")
scores = model.predict(test_df)
```

Откат к предыдущей версии — одной кнопкой в UI, без правки путей.

### 1.3 Сравнение экспериментов

В `CLAUDE.md` в «Дальнейших улучшениях» уже перечислены вариации: без утечки
агрегатов, GRU/Transformer, time-based split, semi-supervised pretrain. Без
MLFlow каждое сравнение — это руками сводить `metrics.json` в табличку. С
ним — три ортогональных ярлыка:

- `mlflow.set_tag("aggregation_scheme", "leaky" | "time_aware")`
- `mlflow.set_tag("architecture", "mlp" | "gru" | "transformer")`
- `mlflow.set_tag("split", "stratified" | "time_holdout")`

И через `mlflow.search_runs(filter_string="tags.architecture='gru'")`
получаешь сравнение.

### 1.4 Что НЕ стоит трогать через MLFlow

- **feature_generator** — детерминирован по seed, артефакты огромные (4 ГБ
  файлы). Логировать parquet'ы в MLFlow артефакты бессмысленно. Логируй
  только **хеш входных файлов** как тег (`data_hash`).
- **`customer_features.parquet`** — то же самое, регистрируй как ссылку, не
  как артефакт.

### 1.5 Минимальный план миграции

1. `pip install mlflow` + `mlflow.set_tracking_uri("file:./mlruns")` в `trainer/train.py`.
2. Обернуть существующий `train()` в `with mlflow.start_run()`, перенести
   `log_params` / `log_metrics` (≈ 10 строк).
3. После сохранения `best.pt` добавить `mlflow.pyfunc.log_model(...)` с
   `FraudPyfunc`.
4. Опционально переписать `predict_example.py` на
   `mlflow.pyfunc.load_model("models:/fraud_mlp/Production")`.
5. `metrics.json` пока оставить — дешёвый локальный fallback.

Главная польза приходит на шаге 3 (несколько вариантов архитектуры) и при
появлении второго разработчика/CI — для одиночного однократного обучения
MLFlow это overkill.

---

## 2. Инструменты для ежедневного переобучения

Нужно покрыть четыре зоны: **расписание**, **версионирование данных и
моделей**, **проверки качества данных/дрейфа** и **инкрементальный пересчёт
фич**.

### 2.1 Оркестратор (расписание + DAG)

Реальный вопрос — не «cron vs не-cron», а «нужно ли видеть прошлые запуски,
ретраи, зависимости между задачами». Задач много (`feature_generator →
aggregate → extract → train → register → deploy`), любая может упасть.

| Инструмент | Когда брать |
|---|---|
| **Prefect 2.x** | По умолчанию для solo/малой команды. Питоновый DAG из декораторов, локальный UI, ретраи и логи из коробки. Минимум boilerplate. |
| **Airflow** | Если уже есть инфраструктура / команда / k8s. Тяжелее, но индустриальный стандарт. |
| **Dagster** | Если хочешь сильную интеграцию с data assets (parquet'ы как «материализованные ассеты» с lineage). |
| **systemd timer / cron** | Только если пайплайн остался ≤ 1 скрипту и устраивает «упало — узнаешь когда придёт паника». |

Под этот проект — **Prefect**. Один файл `daily_flow.py` с `@flow` и
`@task`, расписание `Deployment(... schedule=CronSchedule("0 3 * * *"))`,
retries на каждом шаге.

### 2.2 Версионирование данных

Без этого через месяц непонятно, какой `data_augmented/` соответствует
`fraud_mlp v17`.

- **DVC** — git-like для больших файлов. `dvc add data_augmented/`, в git
  коммитится `.dvc`-указатель. Ложится на текущий parquet-формат, не
  требует БД. Минимальный осмысленный шаг.
- **LakeFS** — git поверх объектного хранилища (S3/MinIO). Мощнее, но нужен
  сервер. Бери если данных > сотен ГБ.
- **Delta Lake / Apache Iceberg** — время-путешествие на уровне таблиц +
  ACID-append. Подходит идеально под «каждый день добавляются новые
  события», но требует Spark / DuckDB / pyarrow-iceberg и переезд с plain
  parquet.

Под этот проект — начни с **DVC**, переезжай на **Iceberg/Delta** когда
инкрементальный append начнёт реально болеть.

### 2.3 Реестр моделей + tracking

**MLFlow Model Registry** (см. раздел 1). Каждый день обучается новая
версия, тегается датой и hash'ом данных:

```python
mlflow.set_tag("data_version", dvc_hash)
mlflow.set_tag("train_date", "2026-05-11")
```

Промоушен **Staging → Production** не должен быть автоматическим —
поставь между ними **validation gate**: новая модель попадает в Production
только если её val_auc ≥ текущей prod − ε.

### 2.4 Проверка данных и дрейф

При ежедневном retrain'е это самое уязвимое место — модель обучится на чём
угодно.

- **Great Expectations** или **Pandera** — schema-checks перед обучением.
  Падает пайплайн если, например, `operaton_amt` стал ≥ 50% NaN или
  `session_id` обвалился до 0%.
- **Evidently AI** — detection дрейфа между «вчерашним» и «сегодняшним»
  распределением + drift самого таргета. Бесплатная open-source часть
  достаточная для отчётов и алертов.
- **Alibi Detect / NannyML** — продвинутее для production-мониторинга, но
  overkill пока нет live-инференса под нагрузкой.

Под этот проект — **Pandera** (легче GE, описывается dataclass-стилем) +
**Evidently** для daily-report'а в `mlruns/`.

### 2.5 Специфические подводные камни именно этого пайплайна

1. **`feature_generator` сейчас переписывает файлы целиком.** Для
   ежедневного режима нужен `--append` режим: обрабатывать только новые
   `event_dttm` и дописывать в parquet (или конвертировать в Delta/Iceberg,
   где append нативный).
2. **Агрегаты `customer_features.parquet` считаются по 108 M строк.**
   10 минут потокового groupby — приемлемо, но честный путь —
   инкрементальное обновление (running mean/std через Welford-формулу +
   дописывание только новых батчей). Альтернатива — **Feast** как feature
   store с материализацией.
3. **Labels приходят с задержкой.** В банковском фроде разметка `target`
   отстаёт на дни-недели (чарджбэки / реакция клиента). «Новые данные
   каждый день» обычно означает **новые события сегодня, но новые метки —
   за окно [сегодня−N, сегодня−M]**. Этот лаг должен быть явным в DAG.
4. **Concept drift во фроде агрессивный.** Стоит сразу планировать не «один
   FraudMLP», а **рейтинг моделей**: champion + challenger, и
   shadow-инференс challenger'а на проде с логированием расхождений.
5. **Обратная совместимость препроцессора.** Если завтра в данных появится
   новая категория `pos_cd`, текущая vocab-схема её схлопнет в `OTHER` —
   корректное поведение, но стоит залогировать через Evidently «новые
   категории встречены X% строк» и алертить при росте.

### 2.6 Минимальный рекомендованный стек

```
┌─ Prefect (расписание + DAG, retry, UI)
├─ DVC (data versioning, git-friendly)
├─ Pandera (schema validation перед обучением)
├─ MLFlow (tracking + Model Registry)
├─ Evidently (drift report после обучения)
└─ systemd unit, запускающий Prefect agent (без k8s)
```

Всё работает на одной машине с RTX 4060. Когда упрётся в данные/нагрузку —
мигрируй хранилище на Iceberg, оркестратор оставь на месте.

### 2.7 Что НЕ брать сразу

- **Kubeflow / MLflow on k8s / Seldon Core** — для команды с
  DevOps-ресурсом. Для одной машины каждый компонент даст +неделя сетапа
  без выигрыша.
- **Feast** — пропусти, пока не появится онлайн-инференс с p99 < 50 ms.
  Текущий batch-предикт по parquet'у не выигрывает от feature store.
- **Полный Spark/Databricks** — 108 M строк прекрасно жуёт pandas +
  pyarrow. Spark начинает выигрывать после ~1B.

---

## 3. Разделение бэкенда и тренировочной таски

**Однозначно выносить отдельно.** Связать их через **реестр моделей** (тот
же MLFlow), а не через общий процесс.

### 3.1 Почему не вместе

| Аспект | Бэкенд | Тренировочный пайплайн |
|---|---|---|
| Аптайм | 24/7, низкая латентность | Batch, может падать и ретраиться |
| Ресурсы | CPU + мало RAM, может без GPU | GPU + 30+ ГБ RAM на этапе агрегатов |
| Длительность процесса | долгоживущий | 15–30 минут раз в сутки |
| Скорость релизов | каждый PR, hotfix'ы | реже, требует ревью данных |
| Доступ к данным | только модель + customer_features | весь датасет, метки, train_labels |
| Критичность сбоя | прод-инцидент | вчерашняя модель продолжает работать |

Если положить ежедневный retrain внутрь бэкенда:

1. **Тренировка съест GPU/CPU/RAM** ровно тогда, когда API должен отвечать.
   На RTX 4060 8 GB это означает OOM или таймауты.
2. **Любая ошибка в `feature_generator` или `aggregate`** уронит сервис
   целиком, хотя инференс к данным никакого отношения не имеет.
3. **Compliance** — у бэкенда фрод-детекции обычно read-only доступ к фичам
   клиента; в тренировке нужен доступ к меткам и сырой истории. Слияние
   расширяет attack surface.
4. **Деплой связан**: чтобы откатить регрессию в API-логике, придётся не
   трогать ничего в тренировочном коде, и наоборот.

### 3.2 Правильный контракт между ними

```
┌─ Training host (Prefect/cron) ──┐         ┌── Backend API (FastAPI/…) ─┐
│  daily flow:                     │         │  on startup:                │
│   feature_generator               │         │   model = mlflow.pyfunc     │
│   aggregate                       │  push   │    .load_model(             │
│   train                           │ ─────►  │     "models:/fraud_mlp/     │
│   validate (champion vs new)      │         │      Production")           │
│   promote → Production            │         │                             │
└──────────────┬───────────────────┘         │  /predict → model.predict() │
               │                              └──────────┬──────────────────┘
               ▼                                         │ logs predictions
        MLFlow Registry  ◄──── prediction logs ──────────┘
        (+ artifact store)         (для drift monitoring)
```

Бэкенду нужно знать всего три вещи:
- URL реестра моделей (`MLFLOW_TRACKING_URI`)
- имя модели и стадия (`fraud_mlp@Production`)
- путь к `customer_features.parquet` (либо тоже тянуть из артефакта той же
  версии модели — рекомендуемый путь, чтобы фичи и модель были синхронны)

### 3.3 Как бэкенд узнаёт о новой модели

Три варианта по возрастанию сложности:

1. **Hot reload по сигналу**: эндпоинт `POST /admin/reload-model`, который
   тренировка дёргает после промоушена. Просто, требует webhook из
   Prefect → API.
2. **Polling раз в N минут**: API сам опрашивает реестр, если в Production
   появилась новая версия — подменяет модель в памяти. Не требует
   webhook'а, но даёт задержку.
3. **Restart-on-deploy**: тренировка не трогает работающий сервис, новая
   модель применяется при следующем деплое API. Самый консервативный,
   годится если хочется human-in-the-loop.

Под банковский фрод — **(1)** + флаг `enable_auto_reload`, чтобы при
необходимости можно было выключить и катить руками.

### 3.4 Когда коллокация всё-таки оправдана

Редко, но бывает:

- **Прототип / dev-стенд для одного разработчика** — пока модель и API
  живут в одном репо и нет прода. Текущий случай (`predict_example.py`
  рядом с `trainer/`), и это нормально для итерации.
- **Online learning** (модель апдейтится на каждом событии) — но это не
  дневной retrain, а другая архитектура.
- **Edge inference** на устройстве с обновлением модели on-device.

Во всех остальных случаях — два изолированных деплоя, один общий реестр,
явный контракт.

### 3.5 Конкретно для текущего репо

Сейчас всё в одной папке: `trainer/`, `feature_generator/`,
`predict_*.py`. На этапе MVP это норм. План разделения:

1. Выделить `inference/` (или новый репозиторий) — туда
   `FraudPyfunc`-wrapper + FastAPI-обвязка вокруг `predict_from_json.py`.
   Зависимости: `torch`, `pyarrow`, `mlflow[skinny]`, без
   `pandas`-агрегатов.
2. Оставить `feature_generator/` + `trainer/` как «training repo».
   Зависимости: всё что есть сейчас + `prefect`, `dvc`, `evidently`.
3. Реестр MLFlow — третий компонент (можно файловый бэкенд на общем диске
   на старте, sqlite-backed `mlflow server` потом).
4. `customer_features.parquet` логируется как артефакт модели в MLFlow →
   бэкенд тянет вместе с весами одной командой `load_model`,
   гарантированно из той же версии.

После этого ежедневная таска физически никогда не пересекается с
прод-сервисом — только через реестр.

---

## 4. Статус реализации (на 2026-05-12)

Документ выше — это **проектное обсуждение** перед началом работы. Ниже
зафиксировано что из него действительно реализовано в репо.

### Сделано

| Раздел в этом документе | Где в коде | Статус |
|---|---|---|
| 1.1 MLFlow tracking | `trainer/train.py:_Tracker` | ✅ опционально через `--mlflow-uri` |
| 1.2 PyFunc + Model Registry | `trainer/pyfunc.py` + `train.py:_Tracker.log_pyfunc` | ✅ |
| 2.1 Prefect daily flow | `orchestration/daily_flow.py` | ✅ |
| 2.1 Cron schedule | `orchestration/deployment.py` (CronSchedule env-driven) | ✅ |
| 2.2 DVC | `.dvc/config`, `dvc.yaml`, `params.yaml` | ✅ установлено + `dvc add data/` + 8× source parquet'ов |
| 2.3 Validation gate | `orchestration/promote.py:validate_and_promote` | ✅ AUC tolerance 0.005 |
| 2.5.2 Инкрементальные агрегаты | `trainer/aggregate.py:CustomerAggregator.save_state/load_state` | ✅ |
| 3.2 Hot-reload контракт | `orchestration/notify.py:notify_backend` | ✅ клиентская часть готова, ждёт endpoint в `AntiFraudMain` |
| 3.5 Inference через MLFlow | `trainer/pyfunc.py:FraudPyfunc` (bundled с `code_paths=["trainer"]`) | ✅ |

### Не сделано (в скоупе follow-up)

| Раздел | Почему отложено |
|---|---|
| 2.4 Pandera / Evidently | Schema gap — отдельный тикет, см. `AntiFraudMain/update.md` |
| 2.5.3 Late-label window | Реализовано в `extract`, но требует `label_dttm` в meta — у текущих меток его нет |
| 2.7 Feast / Spark / k8s | Преждевременно — single-host setup пока тянет |
| 3 Backend split | Server-side: event-sink, `/admin/reload-model` — тикеты в `AntiFraudMain` |

### Известные баги текущей реализации

См. `CLAUDE.md` раздел «Известные баги daily_flow»:
1. Overwrites baseline в `data_augmented/` (нет `--input` в subprocess calls).
2. Падает на test.parquet-only events_glob (нет меток).
3. Training с нуля, не fine-tuning (по дизайну, но стоит задокументировать
   warm-start как option).

### Параллельные документы

- `README.md` — корневой обзор + Quick start.
- `CLAUDE.md` — карта проекта для контекстных ассистентов, gotchas, команды.
- `DATA.md` — детальная схема данных + production-партиции.
- `trainer/README.md` — все CLI флаги, MLFlow интеграция.
- `AntiFraudMain/update.md` — schema gap между backend и trainer.
- `/home/clever/.claude/plans/replicated-sauteeing-tower.md` — план реализации
  (зафиксирован после обсуждения, обновлён по ходу).
