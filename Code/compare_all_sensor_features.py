#!/usr/bin/env python3
"""Compare RF/SVM with peak and frequency features from all six sensor axes.

Keeps the original 48 statistics (including magnitudes) and adds six temporal
features per raw channel, for 84 features total. Baselines are preserved.
"""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import validate_random_forest as validation
import compare_svm as svm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'Code/outputs/all_sensor_features'
BASELINE = ROOT / 'Code/outputs'
ORIGINAL_EXTRACT = validation.rf.extract_features
TEMPORAL = ['dominant_frequency', 'periodicity_strength', 'positive_peak_count',
            'negative_trough_count', 'mean_positive_prominence', 'mean_negative_prominence']


def extract_all_axes(window, fs):
    original_axis = validation.rf.STROKE_AXIS
    try:
        features = {}
        for index, axis in enumerate(validation.rf.ACC_COLUMNS + validation.rf.GYRO_COLUMNS):
            validation.rf.STROKE_AXIS = axis
            axis_features = ORIGINAL_EXTRACT(window, fs)
            if index == 0:
                features.update({key: value for key, value in axis_features.items() if key not in TEMPORAL})
            features.update({f'{axis}_{key}': axis_features[key] for key in TEMPORAL})
        return features
    finally:
        validation.rf.STROKE_AXIS = original_axis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train')
    args = parser.parse_args()
    rf_output, svm_output = OUTPUT / 'rf', OUTPUT / 'svm'
    if args.mode == 'train' and any((p / name).exists() for p,name in [(rf_output,'random_forest.joblib'),(svm_output,'svm.joblib')]):
        raise ValueError('An expanded model already exists; use --mode evaluate to reload')
    validation.rf.extract_features = extract_all_axes
    # Persist feature schema version as part of artifact configuration.
    validation.rf.STROKE_AXIS = 'all_six_axes_v1'
    sys.argv = ['validate_random_forest.py','--mode',args.mode,'--output-dir',str(rf_output)]
    validation.main()
    import report_boundary_aware as boundary
    boundary.OUTPUT = rf_output
    boundary.main()
    svm.RF_OUTPUT = rf_output
    sys.argv = ['compare_svm.py','--mode',args.mode,'--output-dir',str(svm_output)]
    svm.main()
    rows=[]
    for variant, model, folder in [
        ('Baseline (54 features)','RF',BASELINE/'rf_all_labels_validation'),
        ('Baseline (54 features)','SVM',BASELINE/'svm_comparison'),
        ('All-axis temporal features (84)','RF',rf_output),
        ('All-axis temporal features (84)','SVM',svm_output)]:
        scores=json.loads((folder/'metrics.json').read_text())['test']
        rows.append(dict(variant=variant,model=model,**scores))
    table=pd.DataFrame(rows)
    table.to_csv(OUTPUT/'strict_comparison.csv',index=False)
    boundary_rows=[]
    for row,folder in zip(rows,[BASELINE/'rf_all_labels_validation',BASELINE/'svm_comparison',rf_output,svm_output]):
        report=pd.read_csv(folder/'boundary_aware_metrics.csv')
        selected=report[(report.split=='test') & (report.participant=='all') & (report.tolerance_s==1)].iloc[0].to_dict()
        boundary_rows.append(dict(variant=row['variant'],model=row['model'],**selected))
    pd.DataFrame(boundary_rows).to_csv(OUTPUT/'boundary_comparison.csv',index=False)
    lines=['# All-six-channel RF and SVM comparison','',
           'The baseline already used statistics from all six raw sensor channels plus accelerometer/gyroscope magnitudes. This experiment extends peak and frequency features from GYRO_1 alone to ACC_0–2 and GYRO_0–2. It retains 48 statistical features and uses 36 axis-specific temporal features, for 84 total.', '',
           'Labels, inspected coverage, two-second windows with 50% overlap, and the 22/4/4 swimmer split are unchanged. RF uses the same forest settings and selects its cutoff on validation swimmers. SVM scaling is fitted on training swimmers only; the same fixed validation search is used as in the baseline. Both fitted models are saved.', '',
           '## Strict test scores','',
           '| Features | Model | Accuracy | Transition precision | Transition recall | Transition F1 |',
           '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["variant"]} | {r["model"]} | {r["accuracy"]:.1%} | {r["transition_precision"]:.1%} | {r["transition_recall"]:.1%} | {r["transition_f1"]:.3f} |')
    lines += ['','## One-second boundary-aware scores','',
              '| Features | Model | Accuracy | Transition F1 | Retained windows |',
              '|---|---|---:|---:|---:|']
    for r in boundary_rows:
        lines.append(f'| {r["variant"]} | {r["model"]} | {r["accuracy"]:.1%} | {r["transition_f1"]:.3f} | {int(r["windows"])} |')
    lines += ['','Boundary-adjacent windows are excluded, not counted as correct. Strict scores remain the baseline. This is exploratory work using previously inspected test swimmers; model selection uses validation swimmers, but a fresh holdout is needed for a final independent assessment. More features do not guarantee improved generalization. All scores are window-level and neighboring windows overlap.', '',
              'Reload both expanded models without fitting:', '', '```sh', '.venv/bin/python Code/compare_all_sensor_features.py --mode evaluate', '```','',
              'Models and detailed audit, validation search, participant results, and predictions are in rf/ and svm/. Original models and label files are preserved.']
    (OUTPUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(table[['variant','model','accuracy','transition_f1']].to_string(index=False))

if __name__ == '__main__':
    main()
