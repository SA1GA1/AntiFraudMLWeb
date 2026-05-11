# feature_generator

Augments the hackathon banking-fraud dataset with **13 synthetic features**
describing device integrity, behavioural biometrics, and network signals.
Output parquet files can be fed directly to a neural network for training.

## Usage

From the repo root (`/home/clever/Documents/ITParkHackathon`):

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

The CLI streams each source file in row-group batches, so even the 688 MB
train parts run with bounded RAM.

## Determinism

All draws come from a SplitMix64 hash of (key, salt, seed) where the key is
`customer_id`, `session_id`, or `event_id` depending on the feature. Re-running
with the same seed produces a byte-identical parquet file.

## What gets generated

| Column | Type | Key | Behaviour |
|---|---|---|---|
| `attestation_status` | str | customer_id | passed / failed_root / emulator / modified_firmware — biased by risk and `compromised` |
| `app_background_events` | int32 | session_id | count of app backgrounding events; higher under fraud / RDP |
| `clipboard_paste_ratio_mobile` | float32 [0,1] | session_id | share of input that was pasted; higher under fraud |
| `entry_source` | str | event_id | manual / push / deeplink / sms_link |
| `connection_type` | str | session_id | wifi_home / cellular / wifi_public / vpn / tor |
| `touch_typing_rhythm` | float32 | session_id | coefficient of variation of inter-key intervals — bots show near-zero |
| `sim_country_mismatch` | int8 (0/1) | customer_id | SIM country vs IP/timezone mismatch flag |
| `network_rtt_avg` | float32 (ms) | session_id | average RTT to bank server; higher via tunnels |
| `biometric_entry_used` | int8 (0/1) | customer_id + event_id | FaceID/TouchID used; capability stable per customer, usage per event |
| `storage_free_percent` | float32 [0.5, 99.5] | customer_id + event_id | free storage; lower for fraud-y devices |
| `battery_charging_state` | str | event_id (+ existing `battery`) | discharging / charging / full / plugged_24_7 |
| `screen_orientation_changes` | int32 | session_id | count of orientation changes; higher under RDP |
| `debugger_attached` | int8 (0/1) | event_id | strongly elevated under fraud or developer_tools=1 |

## Risk model

Each row gets a *risk score* `r ∈ [0,1]`:

* `target == 1` → `r = 0.85`
* `target == 0` → `r = 0.10`
* unknown (pre-train / pre-test / test) → clamped weighted sum of existing
  on-device flags: `compromised`, `developer_tools`, `web_rdp_connection`,
  `phone_voip_call_state`.

All distributions are linearly interpolated between a "benign" and a "fraud"
parameter set using `r`. Generators stay coherent with existing risk fields
even when no label is available.

## Verification (already passed)

* On `test.parquet` (633 683 rows, no labels): 36 columns out, 0 nulls in
  new columns, identical MD5 across reruns with the same seed.
* On a 300 000-row sample of `train_part_1.parquet` joined to labels:

  | Feature | target=0 mean | target=1 mean |
  |---|---|---|
  | app_background_events | 0.26 | 2.10 |
  | clipboard_paste_ratio_mobile | 0.14 | 0.57 |
  | touch_typing_rhythm | 0.40 | 0.29 |
  | sim_country_mismatch | 0.05 | 0.55 |
  | network_rtt_avg (ms) | 30 | 113 |
  | biometric_entry_used | 0.40 | 0.08 |
  | storage_free_percent | 69 | 25 |
  | screen_orientation_changes | 0.40 | 3.28 |
  | debugger_attached | 0.04 | 0.32 |
  | attestation_status = passed | 90 % | 42 % |

## Layout

```
feature_generator/
├── __init__.py
├── generators.py   # SplitMix64-keyed, vectorised samplers
├── augment.py      # row-group streaming reader/writer
├── cli.py          # argparse entrypoint (python -m feature_generator.cli)
└── README.md
```
