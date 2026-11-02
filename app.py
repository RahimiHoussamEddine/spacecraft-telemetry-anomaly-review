"""Interactive, read-only review of the locked OPSSAT-AD test predictions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
RAW = ROOT / "data/raw"

st.set_page_config(page_title="OPS-SAT Anomaly Review", page_icon="🛰️", layout="wide")


@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    metrics = json.loads((REPORTS / "test_metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(REPORTS / "test_predictions.csv")
    raw = pd.read_csv(RAW / "segments.csv", usecols=["segment", "timestamp", "value"])
    return metrics, predictions, raw


st.title("OPS-SAT Telemetry Anomaly Review")
st.caption(
    "Segment-level benchmark review. The selected model and alert threshold were fixed using "
    "validation data before this locked test set was scored. This tool does not retrain models."
)

try:
    report, predictions, raw_signals = load_artifacts()
except FileNotFoundError:
    st.error("Artifacts are missing. Run `make fetch`, `make train`, and `make evaluate` first.")
    st.stop()

selected = report["selected_model_by_validation_average_precision"]
metrics = report["models"][selected]
threshold = float(report["thresholds"][selected])
first, second, third, fourth = st.columns(4)
first.metric("Validation-selected model", selected.replace("_", " ").title())
second.metric("Test average precision", f"{metrics['average_precision']:.3f}")
third.metric("Test recall at frozen threshold", f"{metrics['recall']:.1%}")
fourth.metric("Test false alerts", f"{metrics['fp']}")
st.caption(f"Frozen alert threshold: {threshold:.6g}. Test size: {metrics['n']} segments.")

left, right = st.columns([1, 2])
with left:
    channels = ["All channels", *sorted(predictions["channel"].unique())]
    channel_filter = st.selectbox("Telemetry channel", channels, key="channel_filter")
    label_filter = st.selectbox(
        "Reference label", ["All segments", "Anomalies", "Nominal segments"], key="label_filter"
    )
    ordering = st.selectbox("Order segments by", ["Highest score", "Lowest score", "Segment ID"])

filtered = predictions.copy()
if channel_filter != "All channels":
    filtered = filtered.loc[filtered["channel"] == channel_filter]
if label_filter == "Anomalies":
    filtered = filtered.loc[filtered["anomaly"] == 1]
elif label_filter == "Nominal segments":
    filtered = filtered.loc[filtered["anomaly"] == 0]
if ordering == "Highest score":
    filtered = filtered.sort_values("selected_score", ascending=False)
elif ordering == "Lowest score":
    filtered = filtered.sort_values("selected_score", ascending=True)
else:
    filtered = filtered.sort_values("segment")

with left:
    if filtered.empty:
        st.warning("No test segments match those filters.")
        st.stop()
    segment_id = st.selectbox(
        "Segment to inspect", filtered["segment"].astype(int).tolist(), key="segment_selector"
    )
    row = predictions.loc[predictions["segment"] == segment_id].iloc[0]
    st.write(
        {
            "channel": row["channel"],
            "reference label": "anomaly" if int(row["anomaly"]) else "nominal",
            "score": float(row["selected_score"]),
            "alert": bool(row["alert"]),
            "sample count": int(row["length"]),
        }
    )
    st.dataframe(
        filtered[["segment", "channel", "anomaly", "selected_score", "alert"]].head(20),
        hide_index=True,
        width="stretch",
    )

with right:
    trace = raw_signals.loc[raw_signals["segment"] == segment_id].copy()
    trace["timestamp"] = pd.to_datetime(trace["timestamp"], utc=True)
    trace = trace.sort_values("timestamp")
    figure = px.line(
        trace,
        x="timestamp",
        y="value",
        title=f"Segment {int(segment_id)} · {row['channel']}",
        labels={"timestamp": "Source timestamp (UTC)", "value": "Telemetry value"},
    )
    figure.update_layout(height=470, margin={"l": 20, "r": 20, "t": 55, "b": 20})
    st.plotly_chart(figure, width="stretch")

st.subheader("Locked test-set model comparison")
comparison = pd.DataFrame(report["models"]).T.reset_index(names="model")
st.dataframe(
    comparison[["model", "average_precision", "roc_auc", "precision", "recall", "f1", "mcc"]],
    hide_index=True,
    width="stretch",
)
st.info(
    "The comparison is a one-time report on the official test split; use validation metrics when "
    "choosing or changing a model. Channel and segment views are descriptive error analysis, not "
    "new model selection."
)
