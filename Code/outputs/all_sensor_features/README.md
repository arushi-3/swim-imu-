# All-six-channel RF and SVM comparison

The baseline already used statistics from all six raw sensor channels plus accelerometer/gyroscope magnitudes. This experiment extends peak and frequency features from GYRO_1 alone to ACC_0–2 and GYRO_0–2. It retains 48 statistical features and uses 36 axis-specific temporal features, for 84 total.

Labels, inspected coverage, two-second windows with 50% overlap, and the 22/4/4 swimmer split are unchanged. RF uses the same forest settings and selects its cutoff on validation swimmers. SVM scaling is fitted on training swimmers only; the same fixed validation search is used as in the baseline. Both fitted models are saved.

## Strict test scores

| Features | Model | Accuracy | Transition precision | Transition recall | Transition F1 |
|---|---|---:|---:|---:|---:|
| Baseline (54 features) | RF | 94.9% | 91.6% | 91.6% | 0.916 |
| Baseline (54 features) | SVM | 92.9% | 84.7% | 93.5% | 0.889 |
| All-axis temporal features (84) | RF | 95.1% | 94.5% | 89.0% | 0.916 |
| All-axis temporal features (84) | SVM | 86.8% | 71.4% | 94.2% | 0.812 |

## One-second boundary-aware scores

| Features | Model | Accuracy | Transition F1 | Retained windows |
|---|---|---:|---:|---:|
| Baseline (54 features) | RF | 97.4% | 0.952 | 467 |
| Baseline (54 features) | SVM | 94.6% | 0.906 | 467 |
| All-axis temporal features (84) | RF | 98.3% | 0.967 | 467 |
| All-axis temporal features (84) | SVM | 87.8% | 0.808 | 467 |

Boundary-adjacent windows are excluded, not counted as correct. Strict scores remain the baseline. This is exploratory work using previously inspected test swimmers; model selection uses validation swimmers, but a fresh holdout is needed for a final independent assessment. More features do not guarantee improved generalization. All scores are window-level and neighboring windows overlap.

Reload both expanded models without fitting:

```sh
.venv/bin/python Code/compare_all_sensor_features.py --mode evaluate
```

Models and detailed audit, validation search, participant results, and predictions are in rf/ and svm/. Original models and label files are preserved.
