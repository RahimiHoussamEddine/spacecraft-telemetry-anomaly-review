"""Derive segment-level features directly from the raw OPS-SAT signal fragments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from telemetry_anomaly.data import RAW_DIR

FEATURE_COLUMNS = [
    "log_length",
    "log_sampling",
    "mean",
    "std",
    "median_abs_deviation",
    "q01",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "q99",
    "range",
    "abs_mean",
    "skew",
    "kurtosis",
    "diff_abs_mean",
    "diff_std",
    "diff_q95_abs",
    "sign_change_rate",
    "lag1_correlation",
    "normalized_slope",
    "zero_fraction",
    "unique_fraction",
    "gap_fraction",
]
META_COLUMNS = ["segment", "channel", "anomaly", "train", "length"]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _segment_features(group: pd.DataFrame, metadata: pd.Series) -> dict[str, float | int | str]:
    values = group["value"].to_numpy(dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Segment {metadata.segment} has no values or non-finite telemetry")
    values = values[np.argsort(pd.to_datetime(group["timestamp"], utc=True).astype("int64"))]
    count = values.size
    center = float(np.median(values))
    spread = float(np.std(values, ddof=0))
    mad = float(np.median(np.abs(values - center)))
    quantiles = np.quantile(values, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
    scale = spread if spread > 0 else 1.0
    centered = (values - float(np.mean(values))) / scale
    skew = float(np.mean(centered**3)) if count > 2 else 0.0
    kurtosis = float(np.mean(centered**4) - 3.0) if count > 3 else 0.0
    differences = np.diff(values)
    if count > 1:
        time_axis = np.linspace(-1.0, 1.0, count)
        slope = float(np.dot(time_axis, values - np.mean(values)) / max(np.var(time_axis), 1e-12))
        left = values[:-1] - float(np.mean(values[:-1]))
        right = values[1:] - float(np.mean(values[1:]))
        denominator = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
        correlation = float(np.dot(left, right) / denominator) if denominator > 0 else 0.0
        sign_change = (
            float(np.mean(np.signbit(differences[1:]) != np.signbit(differences[:-1])))
            if count > 2
            else 0.0
        )
    else:
        slope = 0.0
        correlation = 0.0
        sign_change = 0.0
    times = (
        pd.to_datetime(group["timestamp"], utc=True).astype("int64").to_numpy(dtype=np.float64)
        / 1e6
    )
    ordered_times = np.sort(times)
    time_diffs = np.diff(ordered_times)
    positive_diffs = time_diffs[time_diffs > 0]
    if positive_diffs.size > 1:
        typical_step = float(np.median(positive_diffs))
        gap_fraction = float(np.mean(positive_diffs > 1.5 * typical_step))
    else:
        gap_fraction = 0.0
    return {
        "segment": int(metadata.name),
        "channel": str(metadata.channel),
        "anomaly": int(metadata.anomaly),
        "train": int(metadata.train),
        "length": int(count),
        "log_length": float(np.log1p(count)),
        "log_sampling": float(np.log1p(float(metadata.sampling))),
        "mean": float(np.mean(values)),
        "std": spread,
        "median_abs_deviation": mad,
        "q01": float(quantiles[0]),
        "q10": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q90": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "range": float(np.max(values) - np.min(values)),
        "abs_mean": float(np.mean(np.abs(values))),
        "skew": skew,
        "kurtosis": kurtosis,
        "diff_abs_mean": float(np.mean(np.abs(differences))) if differences.size else 0.0,
        "diff_std": float(np.std(differences)) if differences.size else 0.0,
        "diff_q95_abs": float(np.quantile(np.abs(differences), 0.95)) if differences.size else 0.0,
        "sign_change_rate": sign_change,
        "lag1_correlation": correlation,
        "normalized_slope": _safe_ratio(slope, spread),
        "zero_fraction": float(np.mean(values == 0)),
        "unique_fraction": float(np.unique(values).size / count),
        "gap_fraction": gap_fraction,
    }


def extract_features(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return one raw-signal feature row per benchmark segment, with source metadata."""
    labels_path = raw_dir / "dataset.csv"
    signal_path = raw_dir / "segments.csv"
    labels = pd.read_csv(labels_path)
    raw = pd.read_csv(
        signal_path,
        usecols=[
            "segment",
            "channel",
            "timestamp",
            "value",
            "sampling",
            "anomaly",
            "train",
            "label",
        ],
    )
    if labels["segment"].duplicated().any():
        raise ValueError("Dataset metadata contains duplicate segment identifiers")
    if raw[["segment", "value", "timestamp"]].isna().any().any():
        raise ValueError("Raw signal table has missing segment IDs, values, or timestamps")
    if not set(raw["label"].dropna().unique()).issubset({"anomaly"}):
        raise ValueError("Unexpected value in the source segment label field")
    metadata = labels.set_index("segment")
    groups = raw.groupby("segment", sort=False)
    if set(groups.groups) != set(metadata.index):
        raise ValueError("Raw signal segment IDs and dataset metadata IDs do not match")
    rows: list[dict[str, float | int | str]] = []
    for segment_id, group in groups:
        row = metadata.loc[segment_id]
        if len(group) != int(row["len"]):
            raise ValueError(
                f"Raw row count does not match declared length for segment {segment_id}"
            )
        for column in ("channel", "anomaly", "train", "sampling"):
            if group[column].nunique(dropna=False) != 1 or group[column].iloc[0] != row[column]:
                raise ValueError(f"Raw {column} disagrees with metadata for segment {segment_id}")
        rows.append(_segment_features(group, row))
    result = pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)
    if result[META_COLUMNS + FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature table contains missing values")
    if not np.isfinite(result[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).all():
        raise ValueError("Feature table contains non-finite values")
    return result


def main() -> None:
    features = extract_features()
    output = Path(__file__).resolve().parents[2] / "data/processed/segment_features.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    print(
        f"Wrote {len(features):,} segment rows and {len(FEATURE_COLUMNS)} raw-signal "
        f"features to {output}"
    )


if __name__ == "__main__":
    main()
