"""Fit nominal-only detectors and a supervised reference without opening the test split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from telemetry_anomaly.data import REPORTS_DIR, validate_local
from telemetry_anomaly.features import FEATURE_COLUMNS, extract_features
from telemetry_anomaly.modeling import (
    MODEL_NAMES,
    choose_f1_threshold,
    evaluate_scores,
    feature_frame,
    make_isolation_pipeline,
    make_logistic_pipeline,
    matrix_for_baseline,
    score_pipeline,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "data/processed/segment_features.parquet"
MODEL_PATH = ROOT / "models/model_bundle.joblib"
VALIDATION_PATH = REPORTS_DIR / "validation_metrics.json"
SPLIT_SEED = 2026


def _save_features(features: pd.DataFrame) -> None:
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURE_PATH, index=False)


def _threshold_and_metrics(y: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    threshold, selection = choose_f1_threshold(y, scores)
    return threshold, {**selection, **evaluate_scores(y, scores, threshold)}


def train() -> dict[str, Any]:
    source_manifest = validate_local()
    frame = extract_features()
    _save_features(frame)
    official_train = frame.loc[frame["train"] == 1].reset_index(drop=True)
    official_test = frame.loc[frame["train"] == 0].reset_index(drop=True)
    if (len(official_train), len(official_test)) != (1594, 529):
        raise ValueError(
            "Unexpected official split sizes: "
            f"{len(official_train)} train, {len(official_test)} test"
        )

    train_part, validation = train_test_split(
        official_train,
        test_size=0.20,
        random_state=SPLIT_SEED,
        stratify=official_train["anomaly"],
    )
    train_part = train_part.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    y_train = train_part["anomaly"].to_numpy(dtype=np.int64)
    y_validation = validation["anomaly"].to_numpy(dtype=np.int64)
    nominal = train_part.loc[train_part["anomaly"] == 0].reset_index(drop=True)
    if len(nominal) < 100:
        raise ValueError(
            "Too few nominal training fragments for the planned unsupervised baselines"
        )

    baseline_matrix, validation_matrix, baseline_parameters = matrix_for_baseline(
        nominal, validation
    )
    baseline_validation_scores = np.mean(validation_matrix, axis=1)

    isolation = make_isolation_pipeline()
    isolation.fit(feature_frame(nominal))
    isolation_validation_scores = score_pipeline("isolation_forest", isolation, validation)

    logistic_candidates: list[tuple[float, Any, np.ndarray, float]] = []
    for c_value in (0.1, 1.0, 10.0):
        candidate = make_logistic_pipeline(c_value)
        candidate.fit(feature_frame(train_part), y_train)
        candidate_scores = score_pipeline("supervised_logistic", candidate, validation)
        candidate_ap = float(average_precision_score(y_validation, candidate_scores))
        logistic_candidates.append((candidate_ap, candidate, candidate_scores, c_value))
    logistic_candidates.sort(key=lambda item: (item[0], item[3]), reverse=True)
    logistic_ap, logistic, logistic_validation_scores, logistic_c = logistic_candidates[0]

    validation_scores = {
        "robust_deviation": baseline_validation_scores,
        "isolation_forest": isolation_validation_scores,
        "supervised_logistic": logistic_validation_scores,
    }
    thresholds: dict[str, float] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for name in MODEL_NAMES:
        thresholds[name], metrics[name] = _threshold_and_metrics(
            y_validation, validation_scores[name]
        )
    selected_model = max(MODEL_NAMES, key=lambda name: metrics[name]["average_precision"])

    bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "split_seed": SPLIT_SEED,
        "validation_fraction": 0.20,
        "selected_model": selected_model,
        "thresholds": thresholds,
        "baseline_center": baseline_parameters["center"],
        "baseline_scale": baseline_parameters["scale"],
        "isolation_forest": isolation,
        "supervised_logistic": logistic,
        "logistic_c": logistic_c,
        "logistic_candidate_ap": {str(c): float(ap) for ap, _, _, c in logistic_candidates},
        "source_sha256": {name: info["sha256"] for name, info in source_manifest["files"].items()},
        "validation_segment_ids": validation["segment"].astype(int).tolist(),
        "training_segment_ids": train_part["segment"].astype(int).tolist(),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    report = {
        "protocol": {
            "official_train_segments": int(len(official_train)),
            "official_test_segments_locked": int(len(official_test)),
            "internal_training_segments": int(len(train_part)),
            "internal_validation_segments": int(len(validation)),
            "internal_split_seed": SPLIT_SEED,
            "threshold_objective": "validation F1; ties choose the more conservative threshold",
            "model_selection": (
                "highest validation average precision among the three declared approaches"
            ),
            "test_used_for_selection": False,
            "nominal_only_fit_segments": int(len(nominal)),
        },
        "selected_model": selected_model,
        "selected_logistic_c": float(logistic_c),
        "logistic_candidate_validation_ap": {
            str(c): float(ap) for ap, _, _, c in logistic_candidates
        },
        "validation": {
            name: {**metrics[name], "threshold": thresholds[name]} for name in MODEL_NAMES
        },
    }
    VALIDATION_PATH.parent.mkdir(exist_ok=True)
    VALIDATION_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Model bundle written to {MODEL_PATH}")
    print(f"Computed {baseline_matrix.shape[1]} robust features from {len(frame):,} segments")
    return report


def main() -> None:
    train()


if __name__ == "__main__":
    main()
