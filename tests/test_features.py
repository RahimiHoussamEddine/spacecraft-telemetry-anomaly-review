from __future__ import annotations

import pandas as pd
import pytest

from telemetry_anomaly.features import FEATURE_COLUMNS, extract_features


def _write_minimal_source(raw_dir, values=(1.0, 2.0, 3.0, 4.0)):
    raw_dir.mkdir(parents=True)
    labels = pd.DataFrame(
        [
            {
                "segment": 7,
                "channel": "CADC0872",
                "anomaly": 1,
                "train": 1,
                "len": len(values),
                "sampling": 1,
            }
        ]
    )
    labels.to_csv(raw_dir / "dataset.csv", index=False)
    signal = pd.DataFrame(
        {
            "segment": [7] * len(values),
            "channel": ["CADC0872"] * len(values),
            "timestamp": [f"2024-01-01T00:00:0{i}.000Z" for i in range(len(values))],
            "value": values,
            "sampling": [1] * len(values),
            "anomaly": [1] * len(values),
            "train": [1] * len(values),
            "label": ["anomaly"] * len(values),
        }
    )
    signal.to_csv(raw_dir / "segments.csv", index=False)


def test_raw_signal_features_are_segment_level_and_finite(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_minimal_source(raw_dir)
    result = extract_features(raw_dir)
    assert len(result) == 1
    assert result.loc[0, "segment"] == 7
    assert result.loc[0, "length"] == 4
    assert result.loc[0, "range"] == 3.0
    assert result.loc[0, "anomaly"] == 1
    assert len(FEATURE_COLUMNS) >= 20


def test_metadata_length_mismatch_is_rejected(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_minimal_source(raw_dir)
    labels = pd.read_csv(raw_dir / "dataset.csv")
    labels.loc[0, "len"] = 99
    labels.to_csv(raw_dir / "dataset.csv", index=False)
    with pytest.raises(ValueError, match="row count"):
        extract_features(raw_dir)


def test_constant_segments_produce_finite_shape_statistics(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_minimal_source(raw_dir, values=(0.0, 0.0, 0.0, 0.0))
    result = extract_features(raw_dir)
    assert result[FEATURE_COLUMNS].notna().all().all()
    assert result.loc[0, "std"] == 0
    assert result.loc[0, "lag1_correlation"] == 0
