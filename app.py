"""TETHER Streamlit interface. Uploaded data is processed in memory only."""

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
      .stApp {
        background: radial-gradient(ellipse at 10% 0%, #15263a 0%, #0b111b 48%, #080d15 100%);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }
      .block-container {
        padding-top: 4.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
      }
      .eyebrow {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: #8da9c4;
        letter-spacing: .12em;
        font-size: .75rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
      }
      .hero-title {
        color: #f4f7fb;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin: 0.2rem 0 0.8rem 0;
      }
      .hero-copy {
        color: #a9b8c8;
        max-width: 760px;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
      }
      .mono {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      }
      div[data-testid="stMetric"] {
        background: rgba(16, 28, 44, 0.7);
        border: 1px solid rgba(151, 180, 205, 0.16);
        padding: 0.85rem 1.1rem;
        border-radius: 10px;
        overflow: hidden;
      }
      div[data-testid="stMetricLabel"] p {
        font-size: 0.85rem !important;
        color: #9ab0c4 !important;
        margin-bottom: 0.2rem !important;
      }
      div[data-testid="stMetricValue"] {
        color: #f4f7fb !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
        word-break: break-word !important;
      }
      .stButton button, .stDownloadButton button {
        border-radius: 8px;
        font-weight: 600;
      }
      section[data-testid="stSidebar"] {
        background: #0c1521;
        border-right: 1px solid rgba(151, 180, 205, 0.15);
      }
      .small-note {
        font-size: 0.83rem;
        color: #91a3b6;
        line-height: 1.55;
      }
      [data-testid="stDataFrame"] {
        border: 1px solid rgba(151, 180, 205, 0.15);
        border-radius: 10px;
        overflow: hidden;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the lightweight telemetry model…")
def load_model():
    return train_model()


@st.cache_data(show_spinner=False)
def load_demo_data(filename: str) -> pd.DataFrame:
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
        uploaded_file = st.file_uploader(
            "Telemetry CSV",
            type=["csv"],
            max_upload_size=5,
            help="UTF-8 CSV, up to 5 MB, 20,000 rows and 40 columns.",
        )
    st.divider()
    st.markdown("### Analysis contract")
    st.markdown(
        "<div class='small-note'>\n"
        "<span class='mono'>WINDOW</span> 64 samples<br>\n"
        "<span class='mono'>MIN WINDOW</span> 32 samples<br>\n"
        "<span class='mono'>EXECUTION</span> CPU only<br>\n"
        "<span class='mono'>STORAGE</span> in-memory only\n"
        "</div>",
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

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.metric("Dataset source", source_label)
    st.caption("Rows are kept in original chronological order.")
with col_info2:
    st.metric("Dimensions", f"{len(frame):,} rows × {len(frame.columns)} cols")
    st.caption("Time/index channels excluded from automated selection.")

st.write("")

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
        "Filter findings",
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
        "sensor",
        "window",
        "start_row",
        "end_row",
        "fault_type",
        "status",
        "severity",
        "model_score_pct",
        "method",
        "missing_pct",
        "recommended_check",
    ]
    if view.empty:
        st.info("No findings match this filter.")
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
    chart_data = pd.to_numeric(frame[chart_sensor], errors="coerce").iloc[:2000].to_frame(name=chart_sensor)
    st.line_chart(chart_data, height=260, width="stretch")
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
