# Evaluation methodology

## Data and unit of analysis

The experiment uses OPSSAT-AD v2 from Zenodo record `15108715`. The pinned `dataset.csv` has
2,123 rows, one per telemetry segment; `segments.csv` has the corresponding 303,493 raw readings.
The source metadata and raw records agree on segment ID, channel, split, anomaly label, and row
count for every segment. The raw `label` text field is constant in this release and is ignored;
the binary segment-level `anomaly` field is the reference outcome. Raw-derived features are made
from the values and timestamps without using the anomaly label.

The official source split is preserved: 1,594 development segments and 529 locked test segments.
Within the official development set, a single deterministic stratified split (seed 2026) creates
1,275 fitting rows and 319 validation rows. The test contains 416 nominal and 113 anomalous
segments. The validation threshold and model comparison are fixed before test evaluation. Segment
labels are the evaluation unit; no point adjustment is used.

## Features and models

Twenty-five features summarize each raw fragment: robust and ordinary distribution summaries,
tail quantiles, first-difference variation, turning rate, lag-one correlation, normalized trend,
zero/unique-value fractions, length, sampling metadata, and timestamp-gap fraction. Channel is a
categorical input. Features are deterministic per segment; learned scalers, medians, and models are
fit on fitting rows only. The only cross-row aggregation is model fitting on the training partition.

Three approaches are compared:

1. **Robust deviation:** per-feature median and MAD-based scales learned on nominal fitting rows;
   score is mean absolute standardized deviation.
2. **Isolation Forest:** 250 trees, fixed seed, one CPU worker, fit on nominal fitting rows only.
3. **Supervised logistic regression:** class-balanced logistic regression on all fitting labels.
   `C` is selected from `{0.1, 1, 10}` using validation average precision.

The validation average-precision winner is the reporting choice. Each approach's alert threshold
maximizes validation F1; ties choose the higher threshold. The test scores every model once at those
frozen thresholds. Ranking performance is reported as average precision and ROC-AUC, alongside
threshold metrics, confusion counts, and recall among the top 1%, 5%, and 10% of ranked alerts.
The selected approach has stratified percentile-bootstrap intervals from 500 resamples of test
positives and negatives separately. These intervals describe test sampling uncertainty, not
transfer to a new spacecraft, mission, or operating regime.

## Measured results

Results from `make train` and the one locked `make evaluate` run:

| Approach | Test AP | Test ROC-AUC | Precision | Recall | F1 | MCC | False alerts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Robust deviation | 0.2200 | 0.5507 | 0.2893 | 0.9115 | 0.4392 | 0.2650 | 253 |
| Isolation Forest | 0.2970 | 0.6562 | 0.3140 | 0.9115 | 0.4671 | 0.3130 | 225 |
| Logistic regression (validation winner) | 0.8649 | 0.9540 | 0.7768 | 0.7699 | 0.7733 | 0.7121 | 25 |

The test positive prevalence is 21.36%, the no-skill AP reference for this test composition. For
the validation-selected logistic model, the stratified 95% bootstrap intervals are 0.8094–0.9132
for AP and 0.9329–0.9725 for ROC-AUC. At the frozen threshold, there are 87 true positives, 25
false positives, 26 false negatives, and 391 true negatives among 529 test segments.

Some channel subsets contain too few positives or negatives for stable ranking metrics; the report
suppresses per-channel AP and ROC-AUC unless both classes have at least 10 test segments. Small
channel precision/recall values are descriptive only.

## Limits

- The official split is stratified at segment level, not chronological and not channel-held-out.
- Fragments were curated as interesting telemetry; observed prevalence and scores are not an
  estimate of in-flight anomaly rates.
- The selected logistic model uses labeled anomalies, so it is a supervised benchmark reference,
  not an unsupervised onboard detector.
- The broad gap between supervised and nominal-only methods shows these hand-built features
  discriminate this benchmark split, but does not prove operational generalization.
- Segment-level labels do not express anomaly cause, duration severity, or spacecraft impact.
- No latency, onboard compute, drift, or live telemetry behavior was evaluated.
- This is a review and research workbench, not a flight-qualified alarm system.
