"""Shared split, scoring, and metric helpers for segment-level experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from telemetry_anomaly.features import FEATURE_COLUMNS

MODEL_NAMES = ("robust_deviation", "isolation_forest", "supervised_logistic")


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FEATURE_COLUMNS + ["channel"]].copy()


def matrix_for_baseline(
    train: pd.DataFrame, other: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = train[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    center = np.median(values, axis=0)
    mad_scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    q_scale = (np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)) / 1.349
    std_scale = np.std(values, axis=0)
    scale = np.where(mad_scale > 1e-10, mad_scale, np.where(q_scale > 1e-10, q_scale, std_scale))
    scale = np.where(scale > 1e-10, scale, 1.0)
    return (
        np.abs((values - center) / scale),
        np.abs((other[FEATURE_COLUMNS].to_numpy(dtype=np.float64) - center) / scale),
        {"center": center, "scale": scale},
    )


def robust_score(matrix: np.ndarray) -> np.ndarray:
    return np.mean(matrix, axis=1)


def score_robust(frame: pd.DataFrame, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    return robust_score(np.abs((values - center) / scale))


def make_isolation_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", RobustScaler(quantile_range=(10.0, 90.0)), FEATURE_COLUMNS),
            ("channel", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["channel"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    from sklearn.ensemble import IsolationForest

    return Pipeline(
        [
            ("features", preprocessor),
            (
                "model",
                IsolationForest(
                    n_estimators=250,
                    max_samples="auto",
                    contamination="auto",
                    random_state=2026,
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_logistic_pipeline(c_value: float = 1.0) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), FEATURE_COLUMNS),
            ("channel", OneHotEncoder(handle_unknown="ignore"), ["channel"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=2026,
                ),
            ),
        ]
    )


def choose_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import f1_score, precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        threshold = float(np.nextafter(float(scores.max()), np.inf))
        return threshold, {"threshold": threshold, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    f1 = (2.0 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    maximum = float(f1.max())
    candidates = np.flatnonzero(np.isclose(f1, maximum))
    index = int(candidates[-1])  # prefer the more conservative threshold when F1 ties
    threshold = float(thresholds[index])
    prediction = scores >= threshold
    return threshold, {
        "threshold": threshold,
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
    }


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    prediction = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    top_capacity_recall: dict[str, float] = {}
    n = len(y_true)
    for capacity in (0.01, 0.05, 0.10):
        count = max(1, int(np.ceil(n * capacity)))
        top_indices = np.argsort(scores)[-count:]
        top_capacity_recall[f"top_{int(capacity * 100)}pct_recall"] = float(
            np.sum(y_true[top_indices]) / max(1, np.sum(y_true))
        )
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, prediction))
        if len(np.unique(prediction)) > 1
        else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(n),
        "anomalies": int(np.sum(y_true)),
        **top_capacity_recall,
    }


def score_pipeline(name: str, pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    features = feature_frame(frame)
    if name == "supervised_logistic":
        return pipeline.predict_proba(features)[:, 1]
    return -pipeline.decision_function(features)
