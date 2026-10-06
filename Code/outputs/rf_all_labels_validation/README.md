# Random Forest validation with all labeled coverage

Run October 6, 2026. All 31 labeled recordings from 30 swimmers were included;
only explicitly covered sections were used. Original labels and legacy scripts
were preserved. Duplicate overlapping intervals for swimmer 6 were interpreted
as their union. Rounded endpoints within one sample were accepted.

The existing 54-feature approach was retained: 2-second windows, 50% overlap,
clean label when at least 90% of samples are clean, 100 trees, maximum depth 4,
minimum leaf size 3, balanced class weights, seed 42. Peak/frequency features
use GYRO_1; statistics include all accelerometer and gyroscope axes and magnitudes.

The participant split is 22 train / 4 validation / 4 test. Unlike the older
five-recording experiment, no participant is shared across these groups.
Threshold 0.65 was selected on validation transition F1, with the existing
recall/precision/lower-threshold tie-breaking rules. No test tuning was done.

| Group | Windows | Accuracy | Transition F1 |
|---|---:|---:|---:|
| Training | 2735 | 97.11% | 0.964 |
| Validation | 516 | 99.03% | 0.991 |
| Test | 508 | 94.88% | 0.916 |

Test transition precision and recall are both 91.56%. There were 141 correctly
identified transition windows, 13 missed transition windows, 13 clean windows
classified as transition, and 341 correctly identified clean windows.

| Test swimmer | Accuracy | Transition F1 |
|---|---:|---:|
| 14 | 93.02% | 0.901 |
| 28 | 97.52% | 0.960 |
| 33 | 89.92% | 0.885 |
| 38 | 99.22% | 0.966 |

These are window-level scores against manual labels, not stroke-count accuracy
or transition-event boundary scores. Overlapping windows are correlated; 508
windows are not 508 independent trials. Only four swimmers were held out, and
most coverage is from recording beginnings. Validation scores were used for
threshold selection and training scores are in-sample. Swimmers 19 and 26 have
no clean windows in covered sections; their clean F1 of zero means no class
support, not observed clean-class failure. Timestamp/sensor checks cannot
establish whether manual labels are physiologically correct.

## Commands from the repository root

Audit only:

```sh
.venv/bin/python Code/validate_random_forest.py
```

Reload and evaluate the saved model without training:

```sh
.venv/bin/python Code/validate_random_forest.py --mode evaluate
```

Train a new experiment, preserving this saved model:

```sh
.venv/bin/python Code/validate_random_forest.py --mode train --output-dir Code/outputs/rf_new_experiment
```

Training refuses to overwrite an existing model. Evaluation checks label/split
hashes and feature settings, and refuses changed inputs for this experiment.
The saved artifact includes the model, feature order, threshold, settings,
label hashes, and scikit-learn version. Original raw sensor files must remain
available for evaluation; features are rebuilt but the forest is not refit.
Use this locally generated joblib artifact; loading untrusted joblib files can
execute code.

Files: audit.json and recording_audit.csv contain data checks; metrics.json
contains pooled scores; participant_metrics.csv contains per-swimmer scores;
all_window_predictions.csv contains times, labels, probabilities and predictions;
feature_importance.csv and threshold_results.csv describe the fitted run;
random_forest.joblib and model_metadata.json preserve the trained version.
