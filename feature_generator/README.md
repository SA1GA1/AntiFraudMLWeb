# feature_generator

Augments the source banking dataset with the **71-column web-fraud schema**
defined in `task.md` (browser identity, mouse/keyboard biometrics, network,
device fingerprints, login/trust). The augmented parquet replaces the source
columns — output contains exactly the 71 columns from `task.md`.

## Usage

From the repo root:

```bash
# Augment every *.parquet in data/ → data_augmented/
python3 -m feature_generator.cli

# Restrict to specific files
python3 -m feature_generator.cli --files test.parquet train_part_1.parquet

# Custom paths / seed / batch size
python3 -m feature_generator.cli \
    --input data \
    --output data_augmented \
    --labels data/train_labels.parquet \
    --seed 42 \
    --batch-rows 200000
```

The CLI streams each source file row-group by row-group; even the 688 MB
train parts run with bounded RAM.

## Determinism

All draws come from a SplitMix64 hash of `(key, salt, seed)` where the key is
`customer_id`, `session_id`, or `event_id` depending on the feature. Re-running
with the same seed produces a byte-identical parquet file.

## Schema (71 columns from task.md)

Identity + passthrough from source: `customer_id`, `event_id`, `session_id`,
`event_dttm`, `operaton_amt`, `currency_iso_cd`, `mcc_code`, `pos_cd`,
`browser_language`, `accept_language`.

Browser identity (per-customer stable): `browser_fingerprint`, `user_agent`,
`browser_name`, `browser_version`, `os_type`, `os_version`,
`screen_resolution`, `screen_color_depth`, `system_language`,
`webgl_vendor`, `canvas_fingerprint`, `audio_fingerprint`.

Security/privacy flags (event/session, risk-biased): `is_developer_tools`,
`is_headless_browser`, `is_incognito`, `is_vpn_detected`, `is_proxy_detected`,
`is_tor_detected`.

Network (per-session): `ip_address_hash`, `connection_type`,
`network_rtt_avg_ms`, `asn`, `isp_name`.

Mouse biometrics (per-session, bot bias): `mouse_velocity_avg`,
`mouse_acceleration_avg`, `mouse_jitter_score`, `mouse_linearity_score`.

Click / scroll: `click_duration_avg_ms`, `right_click_count`,
`scroll_velocity_avg`, `double_click_count`.

Keyboard biometrics: `keyboard_typing_speed_median_ms`,
`keyboard_typing_speed_std_dev`, `keyboard_typing_rhythm_cv`.

Form interactions (per-session): `backspace_ratio`, `clipboard_paste_ratio`,
`copy_events_count`, `paste_events_count`, `tab_switch_count`,
`focus_blur_count`, `form_fill_duration_sec`, `idle_time_before_submit_sec`,
`error_correction_ratio`, `hover_time_avg_ms`, `drag_drop_events`,
`resize_events_count`, `zoom_level`.

Session shape: `session_duration_sec`, `pages_visited_count`.

Login / trust (per-customer + per-event): `login_method`,
`failed_login_attempts`, `time_since_last_login_sec`, `is_new_device`,
`is_new_browser`, `device_trust_score`.

Temporal (derived from `event_dttm`): `hour_of_day`, `day_of_week`,
`timezone_offset`.

Transaction enrichment: `merchant_name`, `transaction_type`.

## Risk model

Each row gets a *risk score* `r ∈ [0,1]`:

* `target == 1` → `r = 0.85`
* `target == 0` → `r = 0.10`
* unknown → clipped weighted sum of source on-device flags
  (`compromised`, `developer_tools`, `web_rdp_connection`,
  `phone_voip_call_state`).

Continuous parameters are linearly interpolated between benign and fraud sets
using `r`; binary flags use `Bernoulli(lerp(p_benign, p_fraud, r))`.

## Layout

```
feature_generator/
├── __init__.py
├── generators.py   # SplitMix64-keyed, vectorised samplers (71 columns)
├── augment.py      # row-group streaming reader/writer
├── cli.py          # argparse entrypoint (python -m feature_generator.cli)
└── README.md
```
