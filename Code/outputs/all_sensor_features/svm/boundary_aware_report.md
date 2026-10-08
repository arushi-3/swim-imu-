# Boundary-aware evaluation

This supplementary report excludes windows whose centers are within a stated distance of an internal manual transition boundary. It does not change labels, predictions, or the saved model, and excluded windows are not credited as correct.

The primary descriptive tolerance is 1 second (half the two-second window length). This is an assumption about boundary uncertainty, not a measured annotation error. Scores at 0.5 and 2 seconds show sensitivity; no tolerance was optimized against test results. Coverage endpoints are excluded from the boundary list.

| Center tolerance | Retained test windows | Excluded windows | Accuracy | Transition precision | Transition recall | Transition F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0 s | 508 | 0 | 86.8% | 71.4% | 94.2% | 0.812 |
| 0.5 s | 487 | 21 | 87.3% | 69.0% | 97.0% | 0.806 |
| 1 s | 467 | 41 | 87.8% | 67.8% | 100.0% | 0.808 |
| 2 s | 434 | 74 | 88.0% | 67.1% | 100.0% | 0.803 |

At 1 second, 10 errors are in excluded boundary-adjacent windows. The remaining errors are retained in the interior-window scores.

These scores answer how well the model classifies away from uncertain boundaries; they do not measure boundary detection quality. Keep the original strict test scores as the baseline. Overlapping windows are correlated and only four test swimmers were evaluated. This report was added after reviewing test errors, so it is supplementary rather than a pre-specified primary metric.

See boundary_aware_metrics.csv for every split and swimmer, and boundary_annotated_predictions.csv for each window’s boundary distance and one-second flag.
