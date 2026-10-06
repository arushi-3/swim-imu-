#!/usr/bin/env python3
"""Audit labeled recordings, train/save the existing RF approach, or reload it.

Run without arguments to audit only. Use --mode train to fit and evaluate;
--mode evaluate reloads the saved model without fitting or tuning thresholds.
"""
from pathlib import Path
import argparse
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import TransitionClassifier_multi_file_130s as rf

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / 'Code' / 'labels'
OUTPUT = ROOT / 'Code' / 'outputs' / 'rf_all_labels_validation'


def metrics(frame):
    y, pred = frame['label'], frame['prediction']
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return dict(windows=len(frame), accuracy=float(accuracy_score(y, pred)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                transition_precision=float(precision[0]), transition_recall=float(recall[0]),
                transition_f1=float(f1[0]), clean_f1=float(f1[1]),
                transition_windows=int(support[0]), clean_windows=int(support[1]))


def build(data_dir, output):
    coverage = pd.read_csv(LABELS / 'label_coverage.csv', dtype={'participant': str})
    transitions = pd.read_csv(LABELS / 'transition_ranges.csv', dtype={'participant': str})
    splits = pd.read_csv(LABELS / 'participant_split.csv', dtype={'participant': str})
    errors, warnings, rows, frames = [], [], [], []
    if splits['participant'].duplicated().any(): errors.append('Duplicate participant split assignments')
    if not set(splits['split']).issubset({'train', 'validation', 'test'}): errors.append('Unknown split name')
    split_map = dict(zip(splits['participant'], splits['split']))
    coverage_keys = set(zip(coverage.participant, coverage.file))
    for key in set(zip(transitions.participant, transitions.file)) - coverage_keys:
        errors.append(f'Transitions without coverage: {key}')
    for key, sections in coverage.groupby(['participant', 'file'], sort=False):
        participant, filename = key
        path = data_dir / participant / filename
        ranges = transitions[(transitions.participant == participant) & (transitions.file == filename)]
        try:
            if participant not in split_map: raise ValueError('Missing participant split')
            if Path(filename).name != filename: raise ValueError('Invalid filename')
            raw = pd.read_csv(path)
            required = ['timestamp', *rf.ACC_COLUMNS, *rf.GYRO_COLUMNS]
            numeric = raw[required].apply(pd.to_numeric, errors='coerce')
            if not np.isfinite(numeric.to_numpy()).all(): raise ValueError('Missing or nonfinite sensor/timestamp values')
            time = (numeric.timestamp - numeric.timestamp.iloc[0]) / 1e9
            delta = np.diff(time)
            if len(delta) == 0 or (delta <= 0).any(): raise ValueError('Non-increasing timestamps')
            fs = 1 / np.median(delta)
            if np.max(delta) > 1.5 / fs: raise ValueError('Timestamp gaps exceed 1.5 sample intervals')
            duration = float(time.iloc[-1])
            ordered = sections.sort_values('labeled_start_s')
            previous_end = -1.0
            intervals = []
            for section in ordered.itertuples():
                start, end = float(section.labeled_start_s), float(section.labeled_end_s)
                if not np.isfinite([start, end]).all() or start < 0 or end <= start: raise ValueError('Invalid coverage interval')
                if start <= previous_end: raise ValueError('Overlapping coverage intervals')
                if end > duration + 1 / fs: raise ValueError(f'Coverage end {end} exceeds duration {duration:.3f}')
                if end > duration: warnings.append(f'{participant}/{filename}: coverage end rounded past last sample by {end-duration:.4f}s')
                intervals.append((start, end)); previous_end = end
            last_end = -1.0
            for transition in ranges.sort_values('start_s').itertuples():
                start, end = float(transition.start_s), float(transition.end_s)
                if not np.isfinite([start, end]).all() or start < 0 or end <= start: raise ValueError('Invalid transition interval')
                if start < last_end: warnings.append(f'{participant}/{filename}: overlapping transition ranges; sample labels use their union')
                if not any(start >= a and end <= b + 1 / fs for a, b in intervals): raise ValueError(f'Transition {start}-{end} outside coverage')
                if end > duration + 1 / fs: raise ValueError('Transition extends beyond recording')
                last_end = max(last_end, end)
            df = numeric.copy(); df['time'] = time
            df = rf.add_magnitudes(df)
            df = rf.add_manual_labels(df, list(zip(ranges.start_s, ranges.end_s)))
            recording_frames = []
            for start, end in intervals:
                section_df = df[(df.time >= start) & (df.time <= end)].copy()
                frame = rf.create_windows(section_df, fs, f'{participant}/{filename}', participant, split_map[participant])
                if frame.empty: raise ValueError('Coverage section too short for a window')
                recording_frames.append(frame)
            frame = pd.concat(recording_frames, ignore_index=True)
            if not np.isfinite(frame[rf.get_feature_columns(frame)].to_numpy()).all(): raise ValueError('Nonfinite features')
            frames.append(frame)
            rows.append(dict(participant=participant, recording=filename, split=split_map[participant],
                             sampling_hz=fs, duration_s=duration, windows=len(frame),
                             transition_windows=int((frame.label == 0).sum()), clean_windows=int((frame.label == 1).sum())))
        except (ValueError, KeyError, OSError) as exc:
            errors.append(f'{participant}/{filename}: {exc}')
    missing = set(splits.participant) - set(coverage.participant)
    if missing: errors.append(f'Split participants without coverage: {sorted(missing)}')
    all_features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_features.empty:
        for split in ['train', 'validation', 'test']:
            subset = all_features[all_features.split == split]
            if subset.empty or subset.label.nunique() != 2: errors.append(f'{split} needs both classes')
    report = dict(errors=errors, warnings=warnings, labeled_recordings=len(coverage_keys),
                  participants=int(coverage.participant.nunique()), transition_ranges=len(transitions),
                  split_swimmers=splits.groupby('split').size().to_dict())
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(rows).to_csv(output / 'recording_audit.csv', index=False)
    print(json.dumps(report, indent=2))
    if errors: raise ValueError('Audit failed; review audit.json before training')
    return all_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['audit', 'train', 'evaluate'], default='audit')
    parser.add_argument('--data-dir', type=Path, default=rf.DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    features = build(args.data_dir, output)
    print(features.groupby(['split', 'label']).size().to_string())
    if args.mode == 'audit': return
    columns = rf.get_feature_columns(features)
    artifact = output / 'random_forest.joblib'
    provenance = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [LABELS / 'transition_ranges.csv', LABELS / 'label_coverage.csv', LABELS / 'participant_split.csv']}
    settings = {key: getattr(rf, key) for key in ['WINDOW_SECONDS', 'WINDOW_OVERLAP', 'CLEAN_FRACTION_THRESHOLD', 'NUMBER_OF_TREES', 'MAX_TREE_DEPTH', 'MIN_SAMPLES_LEAF', 'STROKE_AXIS', 'PEAK_PROMINENCE_FACTOR', 'MIN_PEAK_DISTANCE_SECONDS']}
    if args.mode == 'train':
        if artifact.exists(): raise ValueError('Saved model exists; use evaluate or select a new output directory to retrain')
        model = rf.train_classifier(features[features.split == 'train'], columns)
        thresholds = rf.test_probability_thresholds(model, features[features.split == 'validation'], columns)
        threshold = rf.choose_probability_threshold(thresholds)
        thresholds.to_csv(output / 'threshold_results.csv', index=False)
        bundle = dict(model=model, feature_columns=columns, clean_threshold=threshold,
                      label_hashes=provenance, settings=settings, sklearn_version=sklearn.__version__)
        joblib.dump(bundle, artifact)
        rf.create_feature_importance(model, columns).to_csv(output / 'feature_importance.csv', index=False)
    else:
        bundle = joblib.load(artifact)
        if bundle['feature_columns'] != columns or bundle['settings'] != settings: raise ValueError('Feature configuration differs from saved model')
        if bundle['label_hashes'] != provenance: raise ValueError('Labels/splits changed since fitting; use a new output directory for a new experiment')
        model, threshold = bundle['model'], bundle['clean_threshold']
    predictions = rf.predict_all_windows(model, features, columns, threshold)
    predictions.to_csv(output / 'all_window_predictions.csv', index=False)
    features.to_csv(output / 'all_window_features.csv', index=False)
    results = {split: metrics(predictions[predictions.split == split]) for split in ['train', 'validation', 'test']}
    results['note'] = 'Train scores are in-sample. Validation scores were used to select the threshold. Test scores use held-out swimmers; overlapping windows are not independent observations.'
    results['clean_probability_threshold'] = threshold
    (output / 'metrics.json').write_text(json.dumps(results, indent=2))
    participant_results = [dict(participant=p, split=s, **metrics(frame)) for (p, s), frame in predictions.groupby(['participant', 'split'])]
    pd.DataFrame(participant_results).to_csv(output / 'participant_metrics.csv', index=False)
    metadata = {key: value for key, value in bundle.items() if key != 'model'}
    (output / 'model_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
