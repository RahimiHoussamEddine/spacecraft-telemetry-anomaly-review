"""Evaluate validation-selected thresholds once on the official locked test split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from telemetry_anomaly.data import REPORTS_DIR, validate_local
from telemetry_anomaly.features import extract_features
from telemetry_anomaly.modeling import (
    MODEL_NAMES,
    evaluate_scores,
    score_pipeline,
    score_robust,
)
from telemetry_anomaly.train import FEATURE_PATH, MODEL_PATH

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PATH = REPORTS_DIR / "test_metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"


def _bootstrap_intervals(
    y: np.ndarray, scores: np.ndarray, repetitions: int = 500, seed: int = 2026
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not positive.size or not negative.size:
        return {}
    ap_values = np.empty(repetitions)
    auc_values = np.empty(repetitions)
    for index in range(repetitions):
        sampled = np.concatenate(
            [
                rng.choice(negative, size=negative.size, replace=True),
                rng.choice(positive, size=positive.size, replace=True),
            ]
        )
        ap_values[index] = average_precision_score(y[sampled], scores[sampled])
        auc_values[index] = roc_auc_score(y[sampled], scores[sampled])
    return {
        "method": (
            "stratified percentile bootstrap, resampling test positives and negatives separately"
        ),
        "repetitions": repetitions,
        "seed": seed,
        "average_precision_95pct": [float(v) for v in np.quantile(ap_values, [0.025, 0.975])],
        "roc_auc_95pct": [float(v) for v in np.quantile(auc_values, [0.025, 0.975])],
    }


def evaluate() -> dict[str, Any]:
    source_manifest = validate_local()
    bundle = joblib.load(MODEL_PATH)
    feature_file = FEATURE_PATH
    if feature_file.exists():
        features = pd.read_parquet(feature_file)
    else:
        features = extract_features()
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(feature_file, index=False)
    if bundle["source_sha256"] != {
        name: info["sha256"] for name, info in source_manifest["files"].items()
    }:
        raise ValueError("Model bundle was trained against different data files; rerun make train")

    test = features.loc[features["train"] == 0].reset_index(drop=True)
    train_ids = set(bundle["training_segment_ids"]) | set(bundle["validation_segment_ids"])
    if train_ids.intersection(set(test["segment"].astype(int))):
        raise ValueError("Official test segment IDs overlap the development split")
    if len(test) != 529:
        raise ValueError(f"Expected 529 official test segments, found {len(test)}")
    y = test["anomaly"].to_numpy(dtype=np.int64)

    scores = {
        "robust_deviation": score_robust(test, bundle["baseline_center"], bundle["baseline_scale"]),
        "isolation_forest": score_pipeline("isolation_forest", bundle["isolation_forest"], test),
        "supervised_logistic": score_pipeline(
            "supervised_logistic", bundle["supervised_logistic"], test
        ),
    }
    metrics = {
        name: evaluate_scores(y, scores[name], float(bundle["thresholds"][name]))
        for name in MODEL_NAMES
    }
    selected = bundle["selected_model"]
    selected_scores = scores[selected]
    selected_predictions = selected_scores >= float(bundle["thresholds"][selected])
    by_channel: dict[str, dict[str, Any]] = {}
    for channel, indices in test.groupby("channel").groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        channel_y = y[positions]
        channel_scores = selected_scores[positions]
        channel_prediction = selected_predictions[positions]
        tp = int(np.sum((channel_y == 1) & channel_prediction))
        fp = int(np.sum((channel_y == 0) & channel_prediction))
        fn = int(np.sum((channel_y == 1) & ~channel_prediction))
        tn = int(np.sum((channel_y == 0) & ~channel_prediction))
        adequate_rank_support = min(int(np.sum(channel_y == 0)), int(np.sum(channel_y == 1))) >= 10
        by_channel[str(channel)] = {
            "n": int(positions.size),
            "anomalies": int(channel_y.sum()),
            "average_precision": (
                float(average_precision_score(channel_y, channel_scores))
                if adequate_rank_support
                else None
            ),
            "roc_auc": (
                float(roc_auc_score(channel_y, channel_scores)) if adequate_rank_support else None
            ),
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "rank_metrics_suppressed_for_small_support": not adequate_rank_support,
        }
    predictions = test[["segment", "channel", "anomaly", "length"]].copy()
    predictions["selected_model"] = selected
    predictions["selected_score"] = selected_scores
    predictions["threshold"] = float(bundle["thresholds"][selected])
    predictions["alert"] = selected_predictions.astype(int)
    for name in MODEL_NAMES:
        predictions[f"score_{name}"] = scores[name]
        predictions[f"alert_{name}"] = (scores[name] >= float(bundle["thresholds"][name])).astype(
            int
        )
    REPORTS_DIR.mkdir(exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    report = {
        "protocol": {
            "test_segments": int(len(test)),
            "test_anomalies": int(y.sum()),
            "locked_official_test_split": True,
            "test_used_for_model_or_threshold_selection": False,
            "thresholds_from_internal_validation": bundle["thresholds"],
            "test_bootstrap_intervals": "reported only for the validation-selected approach",
        },
        "selected_model_by_validation_average_precision": selected,
        "thresholds": bundle["thresholds"],
        "models": metrics,
        "selected_model_bootstrap": _bootstrap_intervals(y, selected_scores),
        "selected_model_by_channel": by_channel,
        "source_sha256": bundle["source_sha256"],
    }
    EVALUATION_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Predictions (ignored by Git): {PREDICTIONS_PATH}")
    return report


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
