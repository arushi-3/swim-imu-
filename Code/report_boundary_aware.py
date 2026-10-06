#!/usr/bin/env python3
"""Report strict and boundary-excluded RF scores without changing predictions.

A window is boundary-adjacent when its center is within the stated tolerance
of an annotated transition start/end inside a labeled coverage section.
Coverage endpoints are not internal class boundaries. Adjacent windows are
reported separately, not counted as correct. Tolerances are descriptive;
none is selected to optimize the test score.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from validate_random_forest import metrics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'Code/outputs/rf_all_labels_validation'


def score(frame):
    if frame.empty:
        return {'windows': 0, 'accuracy': None, 'transition_f1': None,
                'transition_precision': None, 'transition_recall': None}
    result = metrics(frame)
    # A class without true support has undefined recall; F1 is undefined only
    # if neither truth nor predictions contains that class.
    for label, name in [(0, 'transition'), (1, 'clean')]:
        if not (frame.label == label).any():
            result[name + '_recall'] = None
        if not (frame.prediction == label).any():
            result[name + '_precision'] = None
        if not (frame.label == label).any() and not (frame.prediction == label).any():
            result[name + '_f1'] = None
    return result


def main():
    predictions = pd.read_csv(OUTPUT / 'all_window_predictions.csv', dtype={'participant': str})
    transitions = pd.read_csv(ROOT / 'Code/labels/transition_ranges.csv', dtype={'participant': str})
    coverage = pd.read_csv(ROOT / 'Code/labels/label_coverage.csv', dtype={'participant': str})
    predictions['nearest_boundary_distance_s'] = np.inf
    for recording, windows in predictions.groupby('recording'):
        participant, filename = recording.split('/')
        ranges = transitions[(transitions.participant == participant) & (transitions.file == filename)]
        sections = coverage[(coverage.participant == participant) & (coverage.file == filename)]
        for section in sections.itertuples():
            start, end = section.labeled_start_s, section.labeled_end_s
            boundaries = [float(value) for value in list(ranges.start_s) + list(ranges.end_s) if start < value < end]
            indices = windows.index[windows.center_time.between(start, end)]
            if boundaries and len(indices):
                distances = np.abs(predictions.loc[indices, 'center_time'].to_numpy()[:, None] - np.array(boundaries)).min(axis=1)
                predictions.loc[indices, 'nearest_boundary_distance_s'] = distances
    rows = []
    for split, frame in predictions.groupby('split'):
        groups = [('all', frame)] + list(frame.groupby('participant'))
        for participant, group in groups:
            for tolerance in [0.0, 0.5, 1.0, 2.0]:
                adjacent = group.nearest_boundary_distance_s <= tolerance if tolerance else pd.Series(False, index=group.index)
                interior = group[~adjacent]
                rows.append(dict(split=split, participant=participant, tolerance_s=tolerance,
                                 total_windows=len(group), excluded_boundary_windows=int(adjacent.sum()),
                                 boundary_errors=int((group[adjacent].label != group[adjacent].prediction).sum()),
                                 **score(interior)))
    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT / 'boundary_aware_metrics.csv', index=False)
    predictions['boundary_adjacent_1s'] = predictions.nearest_boundary_distance_s <= 1.0
    predictions.to_csv(OUTPUT / 'boundary_annotated_predictions.csv', index=False)
    pooled = report[(report.split == 'test') & (report.participant == 'all')]
    lines = [
        '# Boundary-aware evaluation', '',
        'This supplementary report excludes windows whose centers are within a stated distance of an internal manual transition boundary. It does not change labels, predictions, or the saved model, and excluded windows are not credited as correct.', '',
        'The primary descriptive tolerance is 1 second (half the two-second window length). This is an assumption about boundary uncertainty, not a measured annotation error. Scores at 0.5 and 2 seconds show sensitivity; no tolerance was optimized against test results. Coverage endpoints are excluded from the boundary list.', '',
        '| Center tolerance | Retained test windows | Excluded windows | Accuracy | Transition precision | Transition recall | Transition F1 |',
        '|---|---:|---:|---:|---:|---:|---:|'
    ]
    for row in pooled.itertuples():
        lines.append(f'| {row.tolerance_s:g} s | {row.windows} | {row.excluded_boundary_windows} | {row.accuracy:.1%} | {row.transition_precision:.1%} | {row.transition_recall:.1%} | {row.transition_f1:.3f} |')
    selected = pooled[pooled.tolerance_s == 1].iloc[0]
    lines += ['', f'At 1 second, {int(selected.boundary_errors)} errors are in excluded boundary-adjacent windows. The remaining errors are retained in the interior-window scores.', '',
              'These scores answer how well the model classifies away from uncertain boundaries; they do not measure boundary detection quality. Keep the original strict test scores as the baseline. Overlapping windows are correlated and only four test swimmers were evaluated. This report was added after reviewing test errors, so it is supplementary rather than a pre-specified primary metric.', '',
              'See boundary_aware_metrics.csv for every split and swimmer, and boundary_annotated_predictions.csv for each window’s boundary distance and one-second flag.']
    (OUTPUT / 'boundary_aware_report.md').write_text('\n'.join(lines) + '\n')
    print(pooled.to_string(index=False))


if __name__ == '__main__':
    main()
