"""Read-only API for aggregate benchmark metrics and individual segment predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
app = FastAPI(
    title="OPS-SAT Telemetry Anomaly Review API",
    version="0.1.0",
    description="Read-only aggregate metrics and held-out segment review records.",
)


def _artifacts() -> tuple[dict[str, Any], pd.DataFrame]:
    report_path = REPORTS / "test_metrics.json"
    predictions_path = REPORTS / "test_predictions.csv"
    if not report_path.exists() or not predictions_path.exists():
        raise HTTPException(status_code=503, detail="Run make train and make evaluate first")
    return (
        json.loads(report_path.read_text(encoding="utf-8")),
        pd.read_csv(predictions_path),
    )


@app.get("/health")
def health() -> dict[str, str]:
    ready = (REPORTS / "test_metrics.json").exists() and (REPORTS / "test_predictions.csv").exists()
    return {"status": "ok" if ready else "not_ready"}


@app.get("/summary")
def summary() -> dict[str, Any]:
    report, _ = _artifacts()
    selected = report["selected_model_by_validation_average_precision"]
    return {
        "selected_model": selected,
        "selection_basis": "internal validation average precision",
        "thresholds_from_validation": report["thresholds"],
        "test_metrics": report["models"],
        "test_protocol": report["protocol"],
    }


@app.get("/segments/{segment_id}")
def segment(segment_id: int) -> dict[str, Any]:
    _, predictions = _artifacts()
    row = predictions.loc[predictions["segment"] == segment_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Segment is not in the official test split")
    return row.iloc[0].to_dict()
