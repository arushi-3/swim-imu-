# SVM comparison with saved Random Forest baseline

Same labeled coverage, 54 features, two-second overlapping windows, and 22/4/4 participant split. StandardScaler and SVM were fitted on training swimmers only. A fixed grid of 12 RBF SVM configurations and five decision thresholds was ranked on validation transition F1, then recall and precision; ties favor lower C and threshold. The selected model was not refitted on validation swimmers. Test results were calculated after selection.

Selected C=1, gamma=0.1, decision threshold=0. Decision scores are margins, not probabilities; they are not directly comparable to the RF probability threshold.

| Model | Boundary tolerance | Test accuracy | Transition precision | Transition recall | Transition F1 | Retained windows |
|---|---|---:|---:|---:|---:|---:|
| Random Forest | 0 s | 94.9% | 91.6% | 91.6% | 0.916 | 508 |
| Random Forest | 1 s | 97.4% | 90.9% | 100.0% | 0.952 | 467 |
| SVM | 0 s | 92.9% | 84.7% | 93.5% | 0.889 | 508 |
| SVM | 1 s | 94.6% | 82.8% | 100.0% | 0.906 | 467 |

One-second scores exclude boundary-adjacent centers; excluded windows are not credited as correct. Strict scores remain the primary baseline. These test swimmers were previously reviewed during RF development, so this is a comparison on an already inspected holdout, not a fresh independent final test. No SVM settings were selected using test scores. Only four swimmers are held out; overlapping windows are correlated.

Reload without retraining from the repository root:

```sh
.venv/bin/python Code/compare_svm.py --mode evaluate
```

The saved pipeline contains both scaling and the SVM. Training refuses to overwrite it. Use --output-dir with a new directory for a separate experiment. RF files and original labels are unchanged.
