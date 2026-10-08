# Each sensor axis tested independently

Each RF or SVM receives exactly 12 features from ONE sensor channel: mean, standard deviation, RMS, minimum, maximum, range, dominant frequency, periodicity strength, positive peak count, negative trough count, and mean positive/negative prominence. Other axes and combined magnitudes are excluded from model inputs. Feature extraction is shared for convenience, but each saved model and prediction file contains only its selected channel features. This directly compares the standalone usefulness of the channels; it does not establish a universally best physical axis.

The same 22 training and four validation swimmers, covered sections and overlapping two-second windows are used throughout. RF settings are fixed. SVM uses the same training-only scaling and fixed C/gamma search for each variant. Thresholds and settings are selected on strict validation transition F1. Test swimmers are not scored or used to rank variants.

| Model | Only input axis | Accuracy | Transition precision | Transition recall | Transition F1 |
|---|---|---:|---:|---:|---:|
| RF | GYRO_1 | 99.2% | 100.0% | 98.6% | 0.993 |
| RF | ACC_0 | 98.6% | 99.3% | 98.3% | 0.988 |
| RF | GYRO_2 | 97.7% | 98.6% | 97.3% | 0.980 |
| RF | GYRO_0 | 96.1% | 98.9% | 94.2% | 0.965 |
| RF | ACC_2 | 95.5% | 98.2% | 93.9% | 0.960 |
| RF | ACC_1 | 95.0% | 99.3% | 91.9% | 0.954 |
| SVM | GYRO_1 | 99.2% | 100.0% | 98.6% | 0.993 |
| SVM | ACC_0 | 98.8% | 99.7% | 98.3% | 0.990 |
| SVM | GYRO_0 | 97.5% | 99.0% | 96.6% | 0.978 |
| SVM | GYRO_2 | 97.3% | 100.0% | 95.3% | 0.976 |
| SVM | ACC_2 | 96.1% | 99.3% | 93.9% | 0.965 |
| SVM | ACC_1 | 95.7% | 99.6% | 92.9% | 0.961 |

## Interpretation

RF: highest validation transition F1 is 0.993; tied variants: GYRO_1.
SVM: highest validation transition F1 is 0.993; tied variants: GYRO_1.

Small differences may reflect these four validation swimmers rather than a robust advantage. Scores are used for model selection and are optimistic estimates of generalization. One validation swimmer has only transition windows. Overlapping windows are correlated. Use participant-level cross-validation within training data to check whether the ranking is consistent. These are reused validation swimmers, and selection scores are not independent final performance estimates. A fresh holdout is preferable for final evaluation. Sensor placement and orientation can change which channel is informative.

Per-swimmer scores, search results, validation predictions and fitted candidate models are saved alongside this report. Existing RF/SVM baselines and labels are preserved. No test accuracy is reported for this axis selection experiment.
