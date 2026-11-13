# Public data validation plan

## Selected dataset: Vehicle CAN bus data (with GPS)

- **Record:** [Zenodo, record 2661316](https://zenodo.org/records/2661316)
- **Dataset title:** Vehicle CAN bus data (with GPS)
- **Creators:** Christian Kaiser, Alexander Stocker, Andreas Festl, Virtual Vehicle Research Center
- **DOI:** [10.5281/zenodo.2661316](https://doi.org/10.5281/zenodo.2661316)
- **License:** CC BY 4.0, as indexed by the NIAID Data Discovery Portal ([catalog record](https://data.niaid.nih.gov/resources?id=ZENODO_2661315)). Verify the current licence and record metadata again before redistributing any derived dataset.
- **Sampling:** the Zenodo description states 20 Hz passenger-vehicle CAN data.
- **Relevant signals:** front-left wheel speed, steering angle, roll / pitch, and accelerometer values per direction. GPS signals are also included.
- **Size:** Zenodo lists 511.1 MB across multiple trip files, with individual HDF files from approximately 29.4 MB upwards. The raw files are intentionally not committed to this repository.

This is a **passenger-vehicle** dataset, not an F1 dataset. It is useful for testing the engineering assumptions around time-series alignment and vehicle-dynamics consistency, but it does not provide labelled sensor-fault windows.

### Attribution

When using it, include the following citation and link to the CC BY 4.0 licence:

> Kaiser, C., Stocker, A., & Festl, A. (2019). *Vehicle CAN bus data (with GPS)*. Zenodo. https://doi.org/10.5281/zenodo.2661316. Licensed under CC BY 4.0.

## Why we do not bundle the data

Raw trip files are much larger than the application and not required to run the synthetic demo. Keeping them outside the repository reduces ZIP size, avoids unnecessary downloads on low-end machines and keeps third-party data clearly separated from MIT-licensed TETHER code. Download only the file needed for a local experiment.

## Bounded HDF conversion

Install the optional reader:

```bash
pip install -r requirements-public-data.txt
```

Export only a small sample first:

```bash
python scripts/convert_vehicle_hdf.py \
  --input external_data/20181114_Driver2_Trip1.hdf \
  --output external_data/vehicle_can_preview.csv \
  --sample-rate 20 \
  --max-rows 20000
```

The converter prints the flattened HDF channel names. Inspect them and confirm the measurement units in the dataset documentation / recording metadata before mapping any channel. Do not assume that a signal named `WheelSpeed FL` is already in km/h or that an accelerometer axis has the exact orientation expected by the profile.

After confirming the source path and units, convert to the canonical profile fields. The mapping uses the flattened column path printed by the first run:

```bash
python scripts/convert_vehicle_hdf.py \
  --input external_data/20181114_Driver2_Trip1.hdf \
  --output external_data/vehicle_can.csv \
  --sample-rate 20 \
  --max-rows 20000 \
  --column-map '{"CAN_WheelSpeed FL":"wheel_speed_fl_kph","CAN_AccelX":"longitudinal_accel_mps2"}'
```

The sample mapping above is illustrative: adjust the source keys to the exact flattened names printed for your HDF file. If source units differ from the canonical profile, convert them explicitly before running the analysis. The converter creates `time_s` from the nominal 20 Hz rate only when the source does not include a time channel. Its console output marks this assumption; generated time is not a captured source timestamp.

Use the `Vehicle CAN passenger car · 20 Hz` profile in the dashboard and provide a separate clean trip / clean segment as the trusted baseline. Do not fit the baseline and score the same samples as if they were independent.

## Evaluation methodology

A naturally recorded trip is not a fault-labelled dataset. Therefore:

1. Fit the trusted relationship baseline on a clean training trip or an early clean segment.
2. Keep an independent clean evaluation segment for false-review-rate measurement.
3. Inject one controlled fault at a time into a copy of the held-out data (e.g. accelerometer bias, wheel-speed drift, dropout, stuck-at, packet loss).
4. Keep the injected row indices as the ground-truth labels.
5. Report window-level precision / recall / F1, clean false alerts per hour, and delay from injection start to the first completed flagged window.
6. Repeat across trips, operating conditions, injection magnitudes and random seeds before drawing conclusions.

Run the included experiment:

```bash
python scripts/evaluate_injected_faults.py \
  --baseline external_data/vehicle_can_clean.csv \
  --profile config/channel_profiles_vehicle_can_20hz.json \
  --output reports/vehicle_can_injected_fault_metrics.json
```

The script requires at least 320 rows and a sufficient number of valid samples in every configured relationship. If the chosen public file does not fit the channel map / expected units, it fails rather than making up a result.

## Motorsport dataset option: RACECAR

The [RACECAR Dataset](https://registry.opendata.aws/racecar-dataset/) contains high-speed autonomous Indy racing records from the 2021–22 Indy Autonomous Challenge, with multi-modal data in ROS 2 and nuScenes formats. Its published licence is **CC BY-NC 4.0**, so it is non-commercial only. It can be a useful later source for racing-context exploration, but it is not a drop-in replacement for a labelled sensor-fault dataset and its license should be considered before public or commercial portfolio use.

## Formula 1 data caution

Do not assume that public timing visualisations or unofficial telemetry APIs provide an unrestricted licence for bulk reuse. Formula 1's published guidelines identify results and timing data among its intellectual-property rights and state that third parties need appropriate permission for uses outside the permitted terms. Keep this project independent of team / championship branding and do not scrape or republish protected telemetry without checking rights first.

## Validation status for this release

The HDF converter and vehicle CAN profile are implemented, but this build environment could not fetch the Zenodo binary file. This release therefore does **not** claim to have run the pipeline on the real public HDF recording. Its checked-in metrics report is based on controlled injections into deterministic synthetic telemetry only. Run the workflow above locally and commit the report plus attribution notes only after successful download, unit verification and reproducible evaluation.
