"""TETHER Streamlit interface. Uploaded data is processed in memory only."""

from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tether.analysis import analyze_dataframe
from tether.model import train_model
from tether.reporting import summarize_results
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
      .hero-copy { color: #a9b8c8; max-width: 760px; font-size: 1.02rem; line-height: 1.7; }
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


@st.cache_data(show_spinner=False)
def load_demo_data(filename: str) -> pd.DataFrame:
    # Only fixed, repository-owned filenames are passed to this function.
    if filename not in {"demo_clean.csv", "demo_faults.csv"}:
        raise ValueError("Unknown demo dataset.")
    return pd.read_csv(ROOT / "data" / filename)


with st.sidebar:
    st.markdown("<div class='eyebrow'>TETHER / CONTROL PANEL</div>", unsafe_allow_html=True)
    st.markdown("### Data source")
    source_mode = st.radio(
        "Choose telemetry source",
        ["Clean synthetic lap", "Fault-injected synthetic lap", "Upload a CSV"],
        label_visibility="collapsed",
    )
    uploaded_file = None
    if source_mode == "Upload a CSV":
        uploaded_file = st.file_uploader("Telemetry CSV", type=["csv"], max_upload_size=5, help="UTF-8 CSV, up to 5 MB, 20,000 rows and 40 columns.")
    st.divider()
    st.markdown("### Analysis contract")
    st.markdown(
        "<div class='small-note'>\n        <span class='mono'>WINDOW</span> 64 samples<br>\n        <span class='mono'>MIN WINDOW</span> 32 samples<br>\n        <span class='mono'>EXECUTION</span> CPU only<br>\n        <span class='mono'>STORAGE</span> no upload persistence\n        </div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("<div class='small-note'>Educational engineering prototype.</div>", unsafe_allow_html=True)

st.markdown("<div class='eyebrow'>MOTORSPORT DATA RELIABILITY / PROTOTYPE 01</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-title'>TETHER<span style='color:#4ad6b3'>.</span></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-copy'>Telemetry Health & Event Reliability Engine. A lightweight integrity gate that looks for questionable sensor signals before they become questionable engineering decisions.</div>",
    unsafe_allow_html=True,
)
st.write("")

frame: pd.DataFrame | None = None
source_label = source_mode
source_id = source_mode
load_error: str | None = None

try:
    if source_mode == "Clean synthetic lap":
        frame = load_demo_data("demo_clean.csv")
    elif source_mode == "Fault-injected synthetic lap":
        frame = load_demo_data("demo_faults.csv")
    elif uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        frame = read_telemetry_csv(uploaded_bytes)
        source_label = uploaded_file.name
        source_id = hashlib.sha256(uploaded_bytes).hexdigest()
except TelemetryValidationError as exc:
    load_error = str(exc)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    load_error = f"Could not load this dataset: {exc}"

if load_error:
    st.error(load_error)

if frame is None:
    st.info("Choose a built-in synthetic run or upload a UTF-8 CSV to begin.")
    st.stop()

sensor_options = numeric_sensor_columns(frame)
if not sensor_options:
    st.error("No mostly-numeric sensor columns were found. Include at least one numeric measurement column.")
    st.stop()

with st.container():
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("**DATASET OVERVIEW**")
        st.text(source_label)
        st.caption("All rows are kept in their original order. Time/index columns are excluded from automatic sensor selection.")
    with right:
        st.metric("Rows × columns", f"{len(frame):,} × {len(frame.columns)}")
        st.caption("Synthetic examples are illustrative, not real vehicle data.")

selected_sensors = st.multiselect(
    "Sensor channels to inspect",
    options=sensor_options,
    default=sensor_options[: min(6, len(sensor_options))],
    help="Choose continuous numeric measurement channels. Discrete state channels may be less suitable for this prototype.",
)

analyze_clicked = st.button("Run integrity analysis", type="primary", width="stretch", disabled=not selected_sensors)

if analyze_clicked:
    try:
        model = load_model()
        with st.spinner("Inspecting sensor windows…"):
            results, warnings = analyze_dataframe(frame, selected_sensors, model)
        st.session_state["tether_results"] = results
        st.session_state["tether_warnings"] = warnings
        st.session_state["tether_sensor_selection"] = selected_sensors
        st.session_state["tether_source_label"] = source_label
        st.session_state["tether_source_id"] = source_id
    except (TelemetryValidationError, ValueError, TypeError) as exc:
        st.error(f"Analysis could not be completed: {exc}")

results = st.session_state.get("tether_results")
if results is not None:
    # Never display a result from a different source or different channel selection.
    same_input = (
        st.session_state.get("tether_source_label") == source_label
        and st.session_state.get("tether_source_id") == source_id
        and st.session_state.get("tether_sensor_selection") == selected_sensors
    )
    if not same_input:
        st.session_state.pop("tether_results", None)
        st.session_state.pop("tether_warnings", None)
        results = None

if results is not None:
    summary = summarize_results(results)
    st.write("")
    st.markdown("## Integrity overview")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Sensors inspected", summary["sensors"])
    metric_cols[1].metric("Windows analyzed", summary["windows"])
    metric_cols[2].metric("Alerts", summary["alerts"])
    metric_cols[3].metric("Needs review", summary["review"])

    warnings = st.session_state.get("tether_warnings", [])
    if warnings:
        with st.expander(f"Data quality notes ({len(warnings)})", expanded=True):
            for warning in warnings:
                st.warning(warning)

    st.markdown("## Findings")
    severity_filter = st.radio(
        "Show findings",
        options=["All", "Alerts & review", "Healthy only"],
        index=0,
        horizontal=True,
    )
    if severity_filter == "Alerts & review":
        view = results[results["status"].isin(["ALERT", "REVIEW"])]
    elif severity_filter == "Healthy only":
        view = results[results["status"] == "HEALTHY"]
    else:
        view = results

    display_columns = [
        "sensor", "window", "start_row", "end_row", "fault_type", "status",
        "severity", "model_score_pct", "method", "missing_pct", "recommended_check",
    ]
    if view.empty:
        st.success("No findings match this filter.")
    else:
        st.dataframe(
            view[display_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "model_score_pct": st.column_config.NumberColumn("Model score (%)", format="%.1f"),
                "recommended_check": st.column_config.TextColumn("Suggested engineering check", width="large"),
            },
        )

    st.markdown("## Signal inspection")
    chart_sensor = st.selectbox("Inspect a channel", options=selected_sensors)
    chart_data = pd.to_numeric(frame[chart_sensor], errors="coerce").iloc[:2000].reset_index(drop=True)
    st.line_chart(chart_data, height=250, width="stretch", alt=f"Time series of {chart_sensor}")
    st.caption("Chart is limited to the first 2,000 samples for responsiveness. Row order is preserved; no signal is resampled.")

    st.markdown("## How to read the score")
    st.markdown(
        "- **Healthy:** the model classified the window as normal with a score of at least 55%.\n"
        "- **Alert:** a fault class was predicted with a score of at least 55%, or a deterministic integrity rule fired.\n"
        "- **Review:** all `DRIFT` findings stay review-only because a single-channel trend is ambiguous; other classes also go to review when the winning score is below 55%.\n\n"
        "The model score is the winning class score from the random forest, **not a calibrated probability** and not a statement that the vehicle is unsafe. Rule-based dropout/stuck findings have no model score."
    )

st.divider()
st.markdown(
    "<div class='small-note'>TETHER is a decision-support prototype. It does not diagnose a mechanical failure, replace engineering judgement, or validate safety-critical telemetry. Use public or synthetic data only in this hosted demo.</div>",
    unsafe_allow_html=True,
)
