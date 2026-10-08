# Boundary-aware evaluation

This supplementary report excludes windows whose centers are within a stated distance of an internal manual transition boundary. It does not change labels, predictions, or the saved model, and excluded windows are not credited as correct.

The primary descriptive tolerance is 1 second (half the two-second window length). This is an assumption about boundary uncertainty, not a measured annotation error. Scores at 0.5 and 2 seconds show sensitivity; no tolerance was optimized against test results. Coverage endpoints are excluded from the boundary list.

| Center tolerance | Retained test windows | Excluded windows | Accuracy | Transition precision | Transition recall | Transition F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0 s | 508 | 0 | 82.5% | 63.7% | 98.1% | 0.772 |
| 0.5 s | 487 | 21 | 82.1% | 60.6% | 99.2% | 0.752 |
| 1 s | 467 | 41 | 82.7% | 59.7% | 100.0% | 0.748 |
| 2 s | 434 | 74 | 82.9% | 58.9% | 100.0% | 0.741 |

At 1 second, 8 errors are in excluded boundary-adjacent windows. The remaining errors are retained in the interior-window scores.

These scores answer how well the model classifies away from uncertain boundaries; they do not measure boundary detection quality. Keep the original strict test scores as the baseline. Overlapping windows are correlated and only four test swimmers were evaluated. This report was added after reviewing test errors, so it is supplementary rather than a pre-specified primary metric.

See boundary_aware_metrics.csv for every split and swimmer, and boundary_annotated_predictions.csv for each window’s boundary distance and one-second flag.
