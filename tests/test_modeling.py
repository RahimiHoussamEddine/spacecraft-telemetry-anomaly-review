from __future__ import annotations

import numpy as np
import pandas as pd

from telemetry_anomaly.features import FEATURE_COLUMNS
from telemetry_anomaly.modeling import choose_f1_threshold, evaluate_scores, score_robust


def test_validation_threshold_and_report_metrics():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    threshold, selection = choose_f1_threshold(y, scores)
    metrics = evaluate_scores(y, scores, threshold)
    assert 0.3 < threshold <= 0.6
    assert selection["f1"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["tp"] == 3
    assert metrics["fp"] == 0


def test_robust_score_is_invariant_to_row_order():
    values = np.arange(20, dtype=float).reshape(4, 5)
    train = pd.DataFrame(values, columns=FEATURE_COLUMNS[:5])
    for column in FEATURE_COLUMNS[5:]:
        train[column] = np.arange(4, dtype=float)
    train["channel"] = ["A", "A", "B", "B"]
    center = np.zeros(len(FEATURE_COLUMNS))
    scale = np.ones(len(FEATURE_COLUMNS))
    scores = score_robust(train, center, scale)
    assert scores.shape == (4,)
    assert np.all(scores >= 0)
