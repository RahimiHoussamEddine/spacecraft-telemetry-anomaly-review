# Spacecraft Telemetry Anomaly Review Workbench

A CPU-friendly benchmark and review interface for detecting anomalous segments in ESA OPS-SAT
telemetry. The workbench starts from the raw variable-length signal fragments, derives its own
segment features, compares nominal-only anomaly detectors with a supervised reference, and
evaluates alerts on the benchmark's locked official test split.

The aim is to make the trade-offs visible to a telemetry analyst: signal gaps and unusual shapes,
false alarms, per-channel variation, and the effect of alert thresholds. This is segment-level
research on a curated benchmark, not a flight-qualified alarm system.

## Dataset and license

The source is [OPSSAT-AD v2 on Zenodo](https://zenodo.org/records/15108715), DOI
`10.5281/zenodo.15108715`. The record includes raw telemetry segments, extracted reference
features, a train/test split, and a bundled MIT license. Retain the attribution `Copyright (c) 2024 KP Labs`;
see [NOTICE.md](NOTICE.md) and [data/OPSSAT_LICENSE.txt](data/OPSSAT_LICENSE.txt). The [benchmark
paper](https://arxiv.org/abs/2407.04730) describes 2,123 manually curated single-channel segments
over nine OPS-SAT telemetry channels, with variable lengths, sampling rates, and real signal gaps.
Three spacecraft operations engineers supplied anomaly annotations, with additional ML expert
curation.

The source files are about 19.5 MB combined. `make fetch` downloads the pinned Zenodo v2 files and
checks the record's MD5 values plus locally computed SHA-256. Raw data remain under `data/raw/` and
are ignored by Git.

## Evaluation design

The project preserves the source benchmark's official segment split: 1,594 training fragments and
529 test fragments, confirmed from the pinned source CSV. A further stratified split of the training
portion supplies an internal validation set; test labels are not used for model or threshold
selection. The comparison is:

- A robust nominal-only baseline built from median and MAD-scaled segment features.
- Isolation Forest trained using nominal training segments only.
- A supervised logistic model as a labeled reference, trained on the same training partition.

The logistic `C` parameter is selected from three values using validation average precision.
Thresholds maximize validation F1, with ties resolved toward the higher (more conservative)
threshold. The locked test comparison reports average precision, ROC-AUC, precision, recall, F1,
MCC, recall at alert-capacity budgets, and per-channel results. Each segment has its own
ground-truth label, so evaluation is segment-level and does not apply point adjustment.

The benchmark itself has limitations: the split is stratified rather than chronological or
channel-held-out, segments were curated as interesting cases, and the fragment-level label does not
identify the operational cause or severity. Results cannot establish flight readiness or general
performance on unseen missions.

## Setup and run

Requirements: Python 3.12 and public Internet access for the licensed source files and packages.
Run these commands from this project directory (the lock uses Python 3.12 exact pins):

```bash
python3.12 -m venv .venv
make install-lock
make fetch
make validate-data
make train
make evaluate
make test
make quality
make smoke-app
```

Start the interactive dashboard:

```bash
make app
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501).

The optional aggregate API runs in another terminal:

```bash
make api
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/summary
curl http://127.0.0.1:8000/segments/1276
```

API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Docker is not
included because a container adds little value to this local benchmark and review workflow.

## Project map

- `src/telemetry_anomaly/data.py` — pinned acquisition and source-file verification.
- `src/telemetry_anomaly/features.py` — raw-segment feature extraction and integrity checks.
- `src/telemetry_anomaly/train.py` — normal-only detectors and validation selection.
- `src/telemetry_anomaly/evaluate.py` — locked test metrics and aggregate reports.
- `app.py` — interactive signal and alert review.
- `src/telemetry_anomaly/api.py` — aggregate evaluation API.
- `data/source_manifest.json` — version, source URLs, expected file hashes, and rights.
- `reports/` — committed aggregate reports; row-level outputs are ignored.

## Measured results

The validation-selected supervised logistic reference reached **0.8649 average precision** and
**0.9540 ROC-AUC** on the 529-segment locked official test split. At the threshold chosen on
validation for maximum F1, it achieved 0.7768 precision, 0.7699 recall, 0.7733 F1, and 0.7121 MCC
(87 true positives, 25 false positives, 26 false negatives, and 391 true negatives). Its
stratified 95% bootstrap interval is 0.8094–0.9132 for average precision and 0.9329–0.9725 for
ROC-AUC. The test positive prevalence is 21.36%.

| Approach | Test AP | ROC-AUC | Precision | Recall | F1 | False alerts |
|---|---:|---:|---:|---:|---:|---:|
| Robust deviation | 0.2200 | 0.5507 | 0.2893 | 0.9115 | 0.4392 | 253 |
| Isolation Forest | 0.2970 | 0.6562 | 0.3140 | 0.9115 | 0.4671 | 225 |
| Logistic regression | 0.8649 | 0.9540 | 0.7768 | 0.7699 | 0.7733 | 25 |

The source split is stratified by segment rather than time or held-out telemetry channel. Segments
are curated examples, so this result does not estimate in-flight prevalence or generalization to a
new mission. See [the full protocol and limitations](docs/methodology.md).
