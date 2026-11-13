"""TETHER Streamlit dashboard. Uploads and baselines are processed in memory only."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tether.analysis import analyze_dataframe, analyze_with_cross_sensor
from tether.baseline import fit_trusted_baseline
from tether.model import train_model
from tether.profiles import load_profile
from tether.quality import check_operating_envelope, diagnose_frame_timing
from tether.reporting import summarize_results
from tether.streaming import StreamProcessor, TelemetryPacket
from tether.validation import (
    TelemetryValidationError,
    numeric_sensor_columns,
    read_telemetry_csv,
)

st.set_page_config(
    page_title="TETHER | Telemetry Integrity Gate",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp { background: radial-gradient(ellipse at 10% 0%, #15263a 0%, #0b111b 48%, #080d15 100%); }
      html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
      .block-container { padding-top: 2.0rem; max-width: 1440px; }
      .eyebrow { font-family: 'DM Mono', monospace; color: #8da9c4; letter-spacing: .16em; font-size: .72rem; }
      .hero-title { color: #f4f7fb; font-size: clamp(2.2rem, 5vw, 4.2rem); font-weight: 800; letter-spacing: -.06em; line-height: 1.02; margin: .45rem 0 .65rem; }
      .hero-copy { color: #a9b8c8; max-width: 820px; font-size: 1.02rem; line-height: 1.7; }
      .panel { border: 1px solid rgba(151,180,205,.18); background: rgba(14,25,38,.76); border-radius: 16px; padding: 1rem 1.15rem; }
      .mono { font-family: 'DM Mono', monospace; }
      div[data-testid="stMetric"] { background: rgba(16,30,45,.78); border: 1px solid rgba(151,180,205,.16); padding: 13px 16px; border-radius: 13px; }
      div[data-testid="stMetricLabel"] { color: #a9b8c8; }
      div[data-testid="stMetricValue"] { color: #f4f7fb; }
      .stButton button, .stDownloadButton button { border-radius: 10px; font-weight: 700; }
      section[data-testid="stSidebar"] { background: #0c1521; border-right: 1px solid rgba(151,180,205,.15); }
      .small-note { font-size: .83rem; color: #91a3b6; line-height: 1.55; }
      [data-testid="stDataFrame"] { border: 1px solid rgba(151,180,205,.15); border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the lightweight telemetry model…")
def load_model():
    return train_model()


@st.cache_resource(show_spinner=False)
def load_default_profile(profile_filename: str):
    allowed_profiles = {
        "channel_profiles.json",
        "channel_profiles_vehicle_can_20hz.json",
    }
    if profile_filename not in allowed_profiles:
        raise ValueError("Unknown channel profile.")
    return load_profile(ROOT / "config" / profile_filename)


@st.cache_data(show_spinner=False)
def load_demo_data(filename: str) -> pd.DataFrame:
    if filename not in {"demo_clean.csv", "demo_faults.csv", "trusted_baseline.csv"}:
        raise ValueError("Unknown built-in dataset.")
    return pd.read_csv(ROOT / "data" / filename)


with st.sidebar:
    st.markdown("<div class='eyebrow'>TETHER / CONTROL PANEL</div>", unsafe_allow_html=True)
    st.markdown("### Data source")
    profile_options = {
        "Generic track car · 20 Hz": "channel_profiles.json",
        "Vehicle CAN passenger car · 20 Hz": "channel_profiles_vehicle_can_20hz.json",
    }
    selected_profile_label = st.selectbox("Channel profile", options=list(profile_options))
    selected_profile_file = profile_options[selected_profile_label]
    source_mode = st.radio(
        "Choose telemetry source",
        ["Clean synthetic lap", "Fault-injected synthetic lap", "Upload a CSV"],
        label_visibility="collapsed",
    )
    uploaded_file = None
    uploaded_baseline = None
    if source_mode == "Upload a CSV":
        uploaded_file = st.file_uploader(
            "Telemetry CSV", type=["csv"], max_upload_size=5,
            help="UTF-8 CSV, up to 5 MB, 20,000 rows and 40 columns.",
        )
        uploaded_baseline = st.file_uploader(
            "Optional trusted baseline CSV", type=["csv"], max_upload_size=5,
            help="A separate, clean recording from the same vehicle, sensor setup and profile.",
        )
    st.divider()
    st.markdown("### Active contract")
    st.markdown(
        "<div class='small-note'>\n"
        f"<span class='mono'>PROFILE</span> {profile_options[selected_profile_label].replace('.json', '')} v1.0.0<br>"
        "<span class='mono'>SAMPLE RATE</span> nominal 20 Hz<br>"
        "<span class='mono'>WINDOW</span> 64 samples<br>"
        "<span class='mono'>EXECUTION</span> CPU / optional C++17<br>"
        "<span class='mono'>STORAGE</span> no upload persistence\n"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<div class='small-note'>Educational engineering prototype.</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='eyebrow'>MOTORSPORT DATA RELIABILITY / PROTOTYPE 02</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-title'>TETHER<span style='color:#4ad6b3'>.</span></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-copy'>Telemetry Health & Event Reliability Engine. "
    "An incremental, cross-sensor integrity gate that checks whether measurements "
    "remain consistent with a trusted vehicle baseline.</div>",
    unsafe_allow_html=True,
)
st.write("")

frame: pd.DataFrame | None = None
baseline_frame: pd.DataFrame | None = None
source_label = source_mode
source_id = source_mode
load_error: str | None = None

try:
    if source_mode == "Clean synthetic lap":
        frame = load_demo_data("demo_clean.csv")
        baseline_frame = load_demo_data("trusted_baseline.csv")
    elif source_mode == "Fault-injected synthetic lap":
        frame = load_demo_data("demo_faults.csv")
        baseline_frame = load_demo_data("trusted_baseline.csv")
    elif uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        frame = read_telemetry_csv(uploaded_bytes)
        source_label = uploaded_file.name
        source_id = hashlib.sha256(uploaded_bytes).hexdigest()
        if uploaded_baseline is not None:
            baseline_frame = read_telemetry_csv(uploaded_baseline.getvalue())
except TelemetryValidationError as exc:
    load_error = str(exc)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    load_error = f"Could not load this dataset: {exc}"

if load_error:
    st.error(load_error)

if frame is None:
    st.info("Choose a built-in synthetic run or upload a UTF-8 CSV to begin.")
    st.stop()

profile = load_default_profile(selected_profile_file)
sensor_options = numeric_sensor_columns(frame)
if not sensor_options:
    st.error("No mostly-numeric sensor columns were found. Include at least one numeric measurement column.")
    st.stop()

baseline = None
baseline_note = "Cross-sensor checks are unavailable until a matching trusted baseline is provided."
if baseline_frame is not None:
    try:
        baseline = fit_trusted_baseline(baseline_frame, profile, provenance="bundled or user-uploaded trusted baseline")
        if not baseline.relationships:
            baseline_note = "The selected baseline does not contain enough samples for the configured relationships."
            baseline = None
        else:
            baseline_note = "Cross-sensor checks use a separately selected trusted baseline."
    except (ValueError, TypeError) as exc:
        baseline_note = f"Trusted baseline could not be fitted: {exc}"

left, right = st.columns([1.5, 1])
with left:
    st.markdown("**DATASET OVERVIEW**")
    st.text(source_label)
    st.caption("Original row order is preserved. No interpolation, sorting or upload persistence is performed.")
with right:
    st.metric("Rows × columns", f"{len(frame):,} × {len(frame.columns)}")
    st.caption("Synthetic baseline is only appropriate for the bundled synthetic demo.")

if source_mode == "Upload a CSV" and baseline is None:
    st.warning("A separate trusted baseline was not supplied. Single-channel ML still runs, but cross-sensor de-escalation is disabled.")
else:
    st.caption(baseline_note)

selected_sensors = st.multiselect(
    "Sensor channels to inspect",
    options=sensor_options,
    default=sensor_options[: min(7, len(sensor_options))],
    help="Continuous numeric channels are recommended. The profile relationship checks require canonical channel names.",
)

analyze_clicked = st.button("Run integrity analysis", type="primary", width="stretch", disabled=not selected_sensors)
if analyze_clicked:
    try:
        model = load_model()
        with st.spinner("Inspecting sensor windows and channel relationships…"):
            if baseline is not None:
                results, cross_results, warnings = analyze_with_cross_sensor(
                    frame, selected_sensors, model, profile, baseline
                )
            else:
                results, warnings = analyze_dataframe(frame, selected_sensors, model)
                cross_results = pd.DataFrame()
        st.session_state["tether_results"] = results
        st.session_state["tether_cross_results"] = cross_results
        st.session_state["tether_warnings"] = warnings
        st.session_state["tether_sensor_selection"] = selected_sensors
        st.session_state["tether_profile_id"] = profile.profile_id
        st.session_state["tether_source_label"] = source_label
        st.session_state["tether_source_id"] = source_id
    except (TelemetryValidationError, ValueError, TypeError) as exc:
        st.error(f"Analysis could not be completed: {exc}")

results = st.session_state.get("tether_results")
cross_results = st.session_state.get("tether_cross_results", pd.DataFrame())
if results is not None:
    same_input = (
        st.session_state.get("tether_source_label") == source_label
        and st.session_state.get("tether_source_id") == source_id
        and st.session_state.get("tether_sensor_selection") == selected_sensors
        and st.session_state.get("tether_profile_id") == profile.profile_id
    )
    if not same_input:
        st.session_state.pop("tether_results", None)
        st.session_state.pop("tether_cross_results", None)
        st.session_state.pop("tether_warnings", None)
        results = None

if results is not None:
    summary = summarize_results(results)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Sensors inspected", summary["sensors"])
    metric_cols[1].metric("Windows analyzed", summary["windows"])
    metric_cols[2].metric("Alerts", summary["alerts"])
    metric_cols[3].metric("Needs review", summary["review"])

    batch_tab, correlation_tab, stream_tab, contract_tab = st.tabs(
        ["Batch findings", "Cross-sensor correlation", "Live packet replay", "Contracts & API"]
    )
    with batch_tab:
        warnings = st.session_state.get("tether_warnings", [])
        if warnings:
            with st.expander(f"Data quality notes ({len(warnings)})", expanded=True):
                for warning in warnings:
                    st.warning(warning)
        timing = diagnose_frame_timing(frame, profile)
        st.markdown("### Timing and packet diagnostics")
        timing_left, timing_right = st.columns(2)
        timing_left.metric("Observed sample rate", f"{timing['sample_rate_hz'] or 'unknown'} Hz")
        timing_right.metric("Sequence gaps", timing["sequence_gaps"])
        if timing["diagnostics"]:
            st.dataframe(pd.DataFrame(timing["diagnostics"]), width="stretch", hide_index=True)
        else:
            st.success("No timestamp or packet-sequence gaps were detected under the active profile.")

        st.markdown("### Findings")
        severity_filter = st.radio(
            "Show findings", ["All", "Alerts & review", "Healthy only"],
            index=0, horizontal=True,
        )
        if severity_filter == "Alerts & review":
            view = results[results["status"].isin(["ALERT", "REVIEW"])]
        elif severity_filter == "Healthy only":
            view = results[results["status"] == "HEALTHY"]
        else:
            view = results
        display_columns = [
            "sensor", "window", "start_row", "end_row", "fault_type", "status",
            "severity", "model_score_pct", "method", "correlation_guard", "missing_pct",
            "range_violations", "recommended_check",
        ]
        display_columns = [name for name in display_columns if name in view.columns]
        if view.empty:
            st.success("No findings match this filter.")
        else:
            st.dataframe(
                view[display_columns], width="stretch", hide_index=True,
                column_config={
                    "model_score_pct": st.column_config.NumberColumn("Model score (%)", format="%.1f"),
                    "recommended_check": st.column_config.TextColumn("Suggested engineering check", width="large"),
                },
            )

        st.markdown("### Operating envelope")
        envelope = check_operating_envelope(frame, profile)
        if envelope.empty:
            st.success("No configured channel exceeded its declared operating envelope.")
        else:
            st.dataframe(envelope, width="stretch", hide_index=True)

        chart_sensor = st.selectbox("Inspect a channel", options=selected_sensors)
        chart_data = pd.to_numeric(frame[chart_sensor], errors="coerce").iloc[:2000].reset_index(drop=True)
        st.line_chart(chart_data, height=250, width="stretch", alt=f"Time series of {chart_sensor}")
        st.caption("Plots are limited to 2,000 samples for responsiveness.")

    with correlation_tab:
        st.markdown("### Trusted relationship residuals")
        if cross_results.empty:
            st.info("Cross-sensor checks require a matching profile and a separate trusted baseline.")
        else:
            flagged = cross_results[cross_results["status"] == "REVIEW"]
            cc1, cc2 = st.columns(2)
            cc1.metric("Relationships checked", cross_results["relationship"].nunique())
            cc2.metric("Windows needing review", len(flagged))
            st.dataframe(cross_results, width="stretch", hide_index=True)
            st.caption("Residual z-scores are deviation indicators relative to the supplied baseline, not calibrated probabilities or mechanical-failure probabilities.")
        st.markdown("### Baseline discipline")
        st.write("Use a separate, known-good recording from the same vehicle, sensor mapping, units, and operating envelope. Do not fit the baseline on the same suspect session.")

    with stream_tab:
        st.markdown("### Incremental packet ingestion")
        st.write("This replay feeds one packet at a time into a bounded buffer. It does not batch-load the whole file into the processor.")
        max_replay = max(64, min(512, len(frame)))
        replay_default = min(256, max_replay)
        replay_limit = st.slider("Packets to replay", min_value=64, max_value=max_replay, value=replay_default, step=64)
        inject_packet_loss = st.checkbox("Simulate a two-packet sequence gap", value=False)
        if st.button("Replay packet stream", type="primary"):
            processor = StreamProcessor(profile.sample_rate_hz, window_size=64)
            replay_frame = frame.iloc[:replay_limit].copy()
            if inject_packet_loss and len(replay_frame) > 130:
                replay_frame = replay_frame.drop(index=replay_frame.index[128:130])
            stream_diagnostics = []
            stream_findings = []
            completed = 0
            channels = [name for name in numeric_sensor_columns(replay_frame) if name in profile.channels]
            if not channels:
                channels = selected_sensors
            stream_model = load_model()
            for index, row in replay_frame.iterrows():
                seq = int(row[profile.sequence_column]) if profile.sequence_column in replay_frame else int(index)
                timestamp = float(row[profile.timestamp_column]) if profile.timestamp_column in replay_frame else float(index) / profile.sample_rate_hz
                values = {
                    name: (None if pd.isna(row[name]) else float(row[name]))
                    for name in channels
                }
                result = processor.ingest(TelemetryPacket(seq, timestamp, values))
                stream_diagnostics.extend(result.diagnostics)
                if result.completed_window is not None:
                    completed += 1
                    if baseline is not None:
                        window_results, _, _ = analyze_with_cross_sensor(
                            result.completed_window, channels, stream_model, profile, baseline
                        )
                    else:
                        window_results, _ = analyze_dataframe(
                            result.completed_window, channels, stream_model
                        )
                    if not window_results.empty:
                        window_results = window_results.copy()
                        window_results["stream_window"] = completed
                        stream_findings.append(window_results)
            st.session_state["tether_stream_snapshot"] = processor.snapshot()
            st.session_state["tether_stream_diagnostics"] = stream_diagnostics
            st.session_state["tether_stream_windows"] = completed
            st.session_state["tether_stream_findings"] = (
                pd.concat(stream_findings, ignore_index=True) if stream_findings else pd.DataFrame()
            )
        snapshot = st.session_state.get("tether_stream_snapshot")
        if snapshot:
            st.json(snapshot)
            stream_diagnostics = st.session_state.get("tether_stream_diagnostics", [])
            st.metric("Completed windows", st.session_state.get("tether_stream_windows", 0))
            stream_findings = st.session_state.get("tether_stream_findings", pd.DataFrame())
            if not stream_findings.empty:
                st.markdown("#### Findings emitted as each window completed")
                st.dataframe(stream_findings, width="stretch", hide_index=True)
            if stream_diagnostics:
                st.warning(f"{len(stream_diagnostics)} stream diagnostic(s) produced.")
                st.dataframe(pd.DataFrame(stream_diagnostics), width="stretch", hide_index=True)
            else:
                st.success("No sequence or timestamp discontinuities detected in this replay.")
        st.code("python scripts/replay_stream.py --limit 256", language="bash")

    with contract_tab:
        st.markdown("### Versioned contracts")
        st.write("Telemetry JSON contract: `contracts/telemetry.schema.json` · Model card: `contracts/model-card.json` · Profile schema: `1.0.0`.")
        st.markdown("### Authenticated local ingestion API")
        st.write("The API is optional and is not exposed by the Streamlit demo. Install the API extra and configure a high-entropy token before starting it.")
        st.code(
            "pip install -r requirements-api.txt\n"
            "# PowerShell: $env:TETHER_API_TOKEN = (python -c \"import secrets; print(secrets.token_urlsafe(32))\")\n"
            "# macOS/Linux: export TETHER_API_TOKEN=...\n"
            "uvicorn tether.api:app --app-dir src --host 127.0.0.1 --port 8000",
            language="bash",
        )
        st.code(
            "curl -X POST http://127.0.0.1:8000/v1/telemetry/packet \\\n"
            "  -H \"Authorization: Bearer $TETHER_API_TOKEN\" \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"contract_version\":\"1.0.0\",\"packet_seq\":0,\"timestamp_s\":0.0,\"values\":{\"speed_kph\":140.0}}'",
            language="bash",
        )
        st.warning("Do not expose this prototype API directly to the public internet. Add TLS termination, rate limiting, a secret manager, network controls, retention policy and an external security review first.")

st.divider()
st.markdown(
    "<div class='small-note'>TETHER is an engineering decision-support prototype. It does not diagnose a mechanical failure, control a vehicle, replace engineering judgement, or validate safety-critical telemetry. Use synthetic or approved public data only.</div>",
    unsafe_allow_html=True,
)
