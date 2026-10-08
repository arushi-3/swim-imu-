# Which axes add useful peak/frequency features?

Validation-only ablation. Every model keeps the same 48 statistical features from all six sensor channels and magnitudes. Only the source of six peak/frequency features changes. Statistics-only uses 48 features, individual-axis variants use 54, and all-six uses 84. This does not compare raw channels alone or prove a physical axis is universally best.

The same 22 training and four validation swimmers, covered sections and overlapping two-second windows are used throughout. RF settings are fixed. SVM uses the same training-only scaling and fixed C/gamma search for each variant. Thresholds and settings are selected on strict validation transition F1. Test swimmers are not scored or used to rank variants.

| Model | Temporal feature axes | Accuracy | Transition precision | Transition recall | Transition F1 |
|---|---|---:|---:|---:|---:|
| RF | statistics_only | 99.0% | 100.0% | 98.3% | 0.991 |
| RF | ACC_0 | 99.0% | 100.0% | 98.3% | 0.991 |
| RF | ACC_1 | 99.0% | 100.0% | 98.3% | 0.991 |
| RF | ACC_2 | 99.0% | 100.0% | 98.3% | 0.991 |
| RF | GYRO_0 | 99.0% | 100.0% | 98.3% | 0.991 |
| RF | GYRO_1 | 99.0% | 100.0% | 98.3% | 0.991 |
| RF | GYRO_2 | 99.0% | 100.0% | 98.3% | 0.991 |
| RF | all_six | 98.8% | 100.0% | 98.0% | 0.990 |
| SVM | GYRO_0 | 99.6% | 100.0% | 99.3% | 0.997 |
| SVM | statistics_only | 99.4% | 99.7% | 99.3% | 0.995 |
| SVM | ACC_0 | 99.4% | 99.7% | 99.3% | 0.995 |
| SVM | ACC_1 | 99.4% | 100.0% | 99.0% | 0.995 |
| SVM | GYRO_1 | 99.4% | 100.0% | 99.0% | 0.995 |
| SVM | GYRO_2 | 99.4% | 100.0% | 99.0% | 0.995 |
| SVM | ACC_2 | 99.2% | 99.7% | 99.0% | 0.993 |
| SVM | all_six | 99.2% | 100.0% | 98.6% | 0.993 |

## Interpretation

RF: highest validation transition F1 is 0.991; tied variants: statistics_only, ACC_0, ACC_1, ACC_2, GYRO_0, GYRO_1, GYRO_2.
SVM: highest validation transition F1 is 0.997; tied variants: GYRO_0.

Small differences may reflect these four validation swimmers rather than a robust advantage. Scores are used for model selection and are optimistic estimates of generalization. One validation swimmer has only transition windows. Overlapping windows are correlated. Confirm a chosen feature set using participant-level cross-validation within training data or a fresh holdout before claiming a universally best axis.

Per-swimmer scores, search results, validation predictions and fitted candidate models are saved alongside this report. Existing RF/SVM baselines and labels are preserved. No test accuracy is reported for this axis selection experiment.
