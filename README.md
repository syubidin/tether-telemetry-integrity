# TETHER

### Telemetry Health & Event Reliability Engine

> **Trust the signal before trusting the decision.**

TETHER is a lightweight, CPU-first telemetry integrity prototype. It combines a small machine-learning classifier with deterministic checks, trusted cross-sensor baselines, and bounded packet ingestion to identify when a measurement may no longer be reliable.

It is designed as a software-engineering portfolio project in the motorsport / vehicle-data domain. It is **not** an official team tool, does not use private team telemetry, and is not affiliated with Mercedes-AMG PETRONAS or Formula 1.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## The engineering problem

A downstream dashboard can only be as trustworthy as the signals it receives. Sensor dropouts, frozen values, drift, spikes, excess noise, timestamp discontinuities, and packet loss can lead engineers to investigate the wrong thing.

A second problem is **false positives**. Vehicle behaviour is coupled and changes constantly: RPM should vary with road speed and gear, speed derivative should broadly agree with longitudinal acceleration, but throttle and speed are not expected to follow one fixed line. A simple threshold can mistake legitimate vehicle dynamics for a sensor fault.

TETHER therefore uses two layers:

1. **Signal-local screening** — a compact Random Forest plus deterministic dropout / flatline rules. Non-normal ML predictions remain review-only because the scores are not calibrated.
2. **Cross-sensor integrity checks** — compare a channel against a trusted, vehicle-specific relationship baseline. A non-normal ML candidate can be de-escalated only when its configured relationship is healthy; missing evidence never suppresses it.

Only deterministic dropout / stuck rules currently create a hard `ALERT`. Drift, spike, noise and uncertain ML findings stay in `REVIEW` unless cross-sensor evidence reduces the suspicion.

## What is in v1.1.0

| Capability | Implementation |
|---|---|
| Lightweight ML | CPU-only Random Forest, bounded windows, batch inference |
| Channel profiles | Versioned units, sample rate, operating envelopes, expected relationships |
| Trusted baselines | Robust linear fit; RPM–speed fit conditioned on gear |
| Cross-sensor residuals | RPM vs speed by gear; speed derivative vs longitudinal accelerometer, including small lag search |
| Timing diagnostics | Timestamp gaps / regressions, duplicate and missing packet sequence checks |
| Streaming | One-packet-at-a-time ingestion with a bounded in-memory window |
| C++ | Optional C++17 speed/acceleration residual kernel via `ctypes`; NumPy fallback |
| API | Optional FastAPI bearer-token ingestion endpoint, strict version contract and body/channel limits |
| Monitoring | Packet counters, gap/reject counts, bounded p50/p95 ingest-latency snapshot |
| Security / quality | Automated tests, Ruff, Bandit, pip-audit and Gitleaks in GitHub Actions |
| Public data path | Bounded HDF-to-CSV adapter for an appropriately licensed public vehicle CAN dataset |

## Quick start

Python 3.11 is recommended. Python 3.10+ is supported by the package configuration.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 2. Install the app and development dependencies
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

# 3. Start the dashboard
streamlit run app.py
```

Open `http://localhost:8501`.

### Dashboard walkthrough

1. Select the **Generic track car · 20 Hz** profile for the bundled sample data.
2. Run the clean and fault-injected synthetic laps.
3. Open **Cross-sensor correlation** to review residuals against the trusted baseline.
4. Open **Live packet replay** to replay packets incrementally and optionally simulate a sequence gap.
5. For your own approved data, select **Upload a CSV**, choose the matching profile, and upload a *separate known-good baseline* from the same vehicle and sensor configuration.

The **Vehicle CAN passenger car · 20 Hz** profile is intended for the public CAN dataset workflow below. The bundled generic synthetic CSV does not contain that profile's channels.

## Example engineering findings

The active demo profiles define these relationships:

- `rpm_vs_speed_conditioned_on_gear`: estimates a separate RPM–speed relationship for each gear, rather than fitting one misleading line across all gears.
- `speed_derivative_vs_accelerometer`: compares the time derivative of road speed (converted from km/h to m/s) with measured longitudinal acceleration. The trusted baseline can select a small sample lag to account for sensor alignment.

Throttle and vehicle speed are deliberately **not** forced into a fixed linear relationship. Driver input, braking, gear changes, traction, and vehicle dynamics make that relationship context-dependent. Adding a simplistic throttle/speed line would likely create false positives.

A high residual means "relationship changed; inspect this window", not "this component has failed". The UI intentionally keeps this state as `REVIEW`.

## Real-time packet replay

Replay the bundled CSV one packet at a time through a bounded buffer:

```bash
python scripts/replay_stream.py --limit 256
```

To send packets over the optional authenticated API, start it in another terminal:

```bash
# Install API dependencies (already included by requirements-dev.txt)
pip install -r requirements-api.txt

# macOS / Linux
export TETHER_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

# Windows PowerShell
$env:TETHER_API_TOKEN = (python -c "import secrets; print(secrets.token_urlsafe(32))")

uvicorn tether.api:app --app-dir src --host 127.0.0.1 --port 8000
```

In a second terminal with the same environment variable:

```bash
python scripts/send_demo_stream.py --limit 128
```

Endpoints:

| Endpoint | Access | Purpose |
|---|---|---|
| `GET /health` | Public | Liveness only; does not reveal internal counters |
| `POST /v1/telemetry/packet` | Bearer token | Validate and ingest one packet |
| `GET /v1/telemetry/metrics` | Bearer token | Read bounded packet / gap / latency counters |

The API accepts telemetry contract `1.0.0`, only configured channel names, at most 40 channels, and request bodies up to 8 KiB. The processor rejects duplicates, out-of-order sequence numbers, timestamp regressions, non-finite measurements and invalid identifiers. Sequence and timestamp gaps are diagnosed without silently sorting or fabricating samples.

## Optional C++17 kernel

The core remains usable without a C++ compiler. Build the native acceleration residual function when CMake and a C++17 compiler are available:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build --output-on-failure
```

The Python adapter discovers the shared library in `build/`. If it is missing or fails to load, TETHER uses the NumPy implementation instead. This keeps the default install portable and low-end friendly.

Check native availability:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from tether.native import native_available; print(native_available())"
```

## Reproducible evaluation

Run the synthetic hold-out classifier evaluation:

```bash
python scripts/evaluate_model.py
```

Run a 30-minute synthetic clean-baseline + controlled fault-injection experiment:

```bash
python scripts/evaluate_injected_faults.py
```

The detailed report is written to `reports/injected_fault_metrics.json`. Current recorded 30-minute synthetic experiment (12-minute held-out evaluation segment):

**Signal-local fault injection**

| Fault | Channel | Precision | Recall | F1 | FP windows | Detection delay |
|---|---|---:|---:|---:|---:|---:|
| DROPOUT | `rpm` | 1.00 | 1.00 | 1.00 | 0 | 3.2 s |
| STUCK | `oil_pressure_bar` | 1.00 | 1.00 | 1.00 | 0 | 3.2 s |
| DRIFT | `coolant_temp_c` | 0.50 | 1.00 | 0.67 | 1 | 3.2 s |
| SPIKE | `throttle_pct` | 1.00 | 1.00 | 1.00 | 0 | 3.2 s |
| NOISE | `speed_kph` | 1.00 | 1.00 | 1.00 | 0 | 3.2 s |

**Cross-sensor residual fault injection**

| Fault | Relationship | Precision | Recall | F1 | Detection delay |
|---|---|---:|---:|---:|---:|
| RPM drift (+500 rpm) | RPM vs speed, conditioned on gear | 1.00 | 1.00 | 1.00 | 3.2 s |
| Accelerometer bias (+1.8 m/s²) | Speed derivative vs longitudinal acceleration | 1.00 | 1.00 | 1.00 | 3.2 s |

**Clean-segment operational counts:** 0.0 hard `ALERT` events/hour, 0.0 cross-sensor `REVIEW` findings/hour, but **3,585 ML review candidates/hour** across all selected numeric channels. The last number is an important unresolved limitation: the synthetic ML model is not calibrated for a low-noise review queue, and the candidate volume is not production-acceptable. Cross-sensor evidence is currently available only for explicitly configured relationships; uncorroborated ML findings remain review-only rather than being silently discarded.

Each fault scenario contains one positive 64-sample target window. Metrics are window-level, not sample-level; the 3.2-second delay is the completion time of a 64-sample window at 20 Hz. The DRIFT scenario also produced one false-positive window. These are controlled software regression results, not robust statistical estimates or accuracy claims for real vehicle faults. See the JSON report for source, split and limitations.

## Public vehicle dataset validation

The repository includes a bounded HDF converter and a profile for the public **Vehicle CAN bus data (with GPS)** dataset. The dataset is passenger-car CAN data sampled at 20 Hz, with signals including front-left wheel speed, steering angle, roll / pitch and accelerometer axes. It is licensed CC BY 4.0 according to the NIAID data catalogue; the original Zenodo record describes the channels and individual trip files. The raw files are deliberately **not bundled**. See [`docs/public_datasets.md`](docs/public_datasets.md) for attribution, caveats and an exact workflow. Release notes are in [`CHANGELOG.md`](CHANGELOG.md).

The converter can export at most 20,000 rows and 40 numeric channels from an HDF file:

```bash
pip install -r requirements-public-data.txt
python scripts/convert_vehicle_hdf.py --input external_data/one_trip.hdf --output external_data/vehicle_can.csv --max-rows 20000
```

Inspect the printed column names and verify units before mapping them to `wheel_speed_fl_kph` and `longitudinal_accel_mps2`. Then use the Vehicle CAN profile and fit a trusted baseline from a separate clean portion / trip. Controlled fault injection can be evaluated with:

```bash
python scripts/evaluate_injected_faults.py \
  --baseline external_data/clean_vehicle_baseline.csv \
  --profile config/channel_profiles_vehicle_can_20hz.json \
  --output reports/vehicle_can_injected_fault_metrics.json
```

**Validation status:** this release includes the licensed-data adapter, profile and evaluation path, but the raw Zenodo file could not be downloaded in the build environment. Consequently, the report currently bundled with the project is synthetic-only; no public-dataset results are claimed yet. The selected dataset is real vehicle data, not Formula 1 telemetry, and does not contain labelled sensor faults. Fault-specific precision / recall still requires controlled injections or separately labelled fault records.

## Engineering constraints and edge cases

- CSV input is bounded to 5 MB, 20,000 rows and 40 columns; API request bodies are bounded to 8 KiB.
- A profile is JSON only, max 64 KiB, with version and channel relationship validation.
- A baseline must be separate from the suspect recording and match profile ID + version.
- Missing channels, too few samples, gear groups below minimum support and invalid timestamps skip the affected relationship rather than inventing a result.
- A missing or mismatched correlation check never suppresses an ML alert.
- Sensor drift is a review candidate; it is not a machine diagnosis.
- Stream buffers are bounded and no uploaded data is persisted by the app code.
- The public Streamlit host still receives uploaded data. Use only synthetic / approved public files in the public demo.

## CI / security checks

GitHub Actions runs the unit and integration tests, Ruff, Bandit, dependency auditing, Gitleaks, and a C++ build/smoke test. Dependabot checks Python and GitHub Actions dependencies weekly. The local dev install includes the scanning tools:

```bash
ruff check app.py src tests scripts
bandit -q -r src scripts app.py
pip-audit -r requirements.txt
pip-audit -r requirements-api.txt
pytest -q
```

## Repository map

```text
.
├── app.py                         # Streamlit dashboard
├── src/tether/                    # ML, feature, baseline, stream, API modules
├── config/                        # Versioned vehicle / channel profiles
├── contracts/                     # Telemetry JSON Schema and model card
├── cpp/                           # Optional C++17 residual kernel
├── data/                           # Small synthetic examples only
├── scripts/                       # Reproducibility, replay, conversion, evaluation
├── reports/                       # Generated synthetic experiment report
├── tests/                         # Python unit / integration tests
└── docs/                           # Architecture, security, dataset and deployment notes
```

## Limitations and intended use

TETHER is an educational research / portfolio prototype. It does not control a vehicle, diagnose mechanical failures, establish a vehicle's safe operating envelope, or replace calibration procedures and engineering review. Its current profiles and synthetic score thresholds must not be used for safety-critical decisions. Real-world use requires vehicle-specific sensor metadata, a representative trusted baseline, calibration, fault labels, deployment hardening, and validation by qualified engineers.

## License

The TETHER source code is MIT licensed. Third-party datasets retain their own licenses and are not included in this repository. See [`docs/public_datasets.md`](docs/public_datasets.md) for dataset attribution.
