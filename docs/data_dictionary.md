# Data dictionary

The downloader retrieves files from the exact OPSSAT-AD v2 Zenodo record named in
`data/source_manifest.json`. It verifies each source MD5 and records the downloaded byte count and
SHA-256 in `reports/data_manifest.json`. The dataset files remain ignored under `data/raw/`.

## `dataset.csv` — one row per segment

The project's modeling outcome is `anomaly`; `train` is the source-provided official split flag
(1 = development, 0 = locked test). `segment` uniquely identifies a fragment. `channel` identifies
one of nine telemetry signals. `len`, `sampling`, and the remaining precomputed statistics are
provided by the source but are not used as raw-derived model features, except for `sampling` as a
single metadata input. The features in this project are recomputed from `segments.csv`.

## `segments.csv` — raw readings

Each row contains `segment`, `channel`, UTC `timestamp`, numeric `value`, sampling metadata, and
repeated segment outcome/split fields. Source validation checks that segment-level values agree
with `dataset.csv`, the raw row count matches the declared length, and timestamps/values are
present. `label` is constant (`anomaly`) in the downloaded v2 file and is not treated as the
segment's ground-truth outcome.

## Derived feature table

`data/processed/segment_features.parquet` contains one row per segment, source metadata, and 25
features computed only from that segment's raw readings. It is a generated local cache and is
excluded from Git. No raw telemetry rows, fitted model binaries, or row-level test predictions are
committed.
