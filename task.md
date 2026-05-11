customer_id — ID клиента банка
event_id — Уникальный ID операции
session_id — ID сессии
device_id — Уникальный идентификатор устройства
installation_id — ID установки приложения (сбрасывается при переустановке)
app_version — Версия мобильного приложения
os_type — Тип ОС (iOS/Android)
os_version — Версия операционной системы
device_model — Модель устройства
event_dttm — Дата и время операции
hour_of_day — Час совершения операции (0-23)
day_of_week — День недели (0-6)
operaton_amt — Сумма операции в рублях
currency_iso_cd — Код валюты операции
mcc_code — MCC-код категории продавца
merchant_name — Название магазина/получателя
pos_cd — Код условия проведения операции
transaction_type — Тип транзакции (p2p/оплата/снятие/перевод)
Аппаратная аттестация и безопасность
attestation_status — Результат аппаратной аттестации устройства
is_rooted_jailbroken — Флаг наличия root/jailbreak
is_emulator — Флаг запуска в эмуляторе
is_debugger_attached — Флаг подключения отладчика
developer_tools_enabled — Включены ли настройки разработчика
app_install_source — Источник установки приложения (официальный/сторонний)
integrity_token — Токен проверки целостности (Play Integrity/DeviceCheck)
connection_type — Тип подключения (wifi/cellular/5g/4g)
carrier_name — Название мобильного оператора
carrier_mcc — Код страны оператора
carrier_mnc — Код сети оператора
ip_address_hash — Хеш IP-адреса
is_vpn_detected — Флаг использования VPN
is_proxy_detected — Флаг использования прокси
network_rtt_avg_ms — Среднее время сетевого отклика
sim_country_code — Код страны SIM-карты
sim_carrier_name — Название оператора SIM
latitude — Широта местоположения
longitude — Долгота местоположения
accuracy_meters — Точность геолокации в метрах
location_provider — Источник геолокации (gps/network/passive)
timezone_offset_minutes — Смещение часового пояса
geo_speed_km_h — Скорость перемещения от последней точки
touch_typing_rhythm_median_ms — Медианное время между касаниями
touch_typing_rhythm_std_dev — Стандартное отклонение скорости ввода
touch_typing_rhythm_cv — Коэффициент вариации ритма ввода
tap_velocity_avg — Средняя скорость нажатий
tap_pressure_avg — Среднее давление нажатия (если доступно)
touch_jitter_score — Оценка дрожания пальца
swipe_angle_deviation — Отклонение угла свайпа
clipboard_paste_ratio — Доля вставок из буфера обмена
backspace_ratio — Доля нажатий клавиши назад
form_fill_duration_sec — Время заполнения формы
app_background_events — Количество сворачиваний приложения
screen_orientation_changes — Количество поворотов экрана
accelerometer_variance_x — Дисперсия акселерометра по оси X
accelerometer_variance_y — Дисперсия акселерометра по оси Y
gyroscope_variance — Дисперсия гироскопа
biometric_entry_used — Флаг входа по биометрии
battery_level — Уровень заряда батареи
battery_charging_state — Статус зарядки устройства
storage_free_percent — Процент свободной памяти