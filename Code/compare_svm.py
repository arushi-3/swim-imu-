#!/usr/bin/env python3
"""Select a scaled RBF SVM on validation swimmers and save an RF comparison.

Training: python Code/compare_svm.py
Reload without fitting: python Code/compare_svm.py --mode evaluate
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from validate_random_forest import build, metrics, LABELS, OUTPUT as RF_OUTPUT, rf
import report_boundary_aware as boundary

OUTPUT = RF_OUTPUT.parent / 'svm_comparison'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    if args.mode == 'train' and (output / 'svm.joblib').exists():
        raise ValueError('Saved SVM exists; evaluate it or choose a new output directory')
    features = build(rf.DEFAULT_DATA_DIR, output)
    columns = rf.get_feature_columns(features)
    hashes = {name: hashlib.sha256((LABELS/name).read_bytes()).hexdigest() for name in ['transition_ranges.csv','label_coverage.csv','participant_split.csv']}
    baseline = joblib.load(RF_OUTPUT / 'random_forest.joblib')
    if baseline['label_hashes'] != hashes or baseline['feature_columns'] != columns:
        raise ValueError('Inputs differ from the RF baseline')
    train = features[features.split == 'train']; validation = features[features.split == 'validation']
    if args.mode == 'train':
        candidates = []; models = {}
        # Small fixed grid; all scaling and fitting use training swimmers only.
        for c in [0.1, 1.0, 10.0, 100.0]:
            for gamma in ['scale', 0.001, 0.1]:
                model = make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, kernel='rbf', class_weight='balanced'))
                model.fit(train[columns], train.label)
                margins = model.decision_function(validation[columns])
                models[(c,str(gamma))] = model
                for threshold in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    frame = validation.copy(); frame['prediction'] = (margins >= threshold).astype(int)
                    candidates.append(dict(C=c, gamma=str(gamma), decision_threshold=threshold, **metrics(frame)))
        search = pd.DataFrame(candidates).sort_values(['transition_f1','transition_recall','transition_precision','C','decision_threshold'], ascending=[False,False,False,True,True], kind='stable')
        search.to_csv(output/'validation_search.csv', index=False)
        best = search.iloc[0]
        model = models[(best.C,best.gamma)]; threshold = float(best.decision_threshold)
        bundle = dict(model=model, feature_columns=columns, label_hashes=hashes,
                      decision_threshold=threshold, C=float(best.C), gamma=best.gamma,
                      sklearn_version=sklearn.__version__, rf_settings=baseline['settings'])
        joblib.dump(bundle, output/'svm.joblib')
    else:
        bundle = joblib.load(output/'svm.joblib')
        if bundle['label_hashes'] != hashes or bundle['feature_columns'] != columns or bundle['rf_settings'] != baseline['settings']:
            raise ValueError('Inputs/settings changed since SVM training')
        model = bundle['model']; threshold = bundle['decision_threshold']
    predictions = features.copy()
    predictions['clean_decision_score'] = model.decision_function(features[columns])
    predictions['prediction'] = (predictions.clean_decision_score >= threshold).astype(int)
    predictions['decision_threshold'] = threshold
    predictions.to_csv(output/'all_window_predictions.csv', index=False)
    scores = {split:metrics(predictions[predictions.split==split]) for split in ['train','validation','test']}
    (output/'metrics.json').write_text(json.dumps(scores,indent=2))
    (output/'model_metadata.json').write_text(json.dumps({k:v for k,v in bundle.items() if k!='model'},indent=2))
    pd.DataFrame([dict(participant=p,split=s,**metrics(f)) for (p,s),f in predictions.groupby(['participant','split'])]).to_csv(output/'participant_metrics.csv',index=False)
    # Identical boundary-distance and scoring definition for both models.
    boundary.OUTPUT = output
    boundary.main()
    svm_boundary = pd.read_csv(output/'boundary_aware_metrics.csv')
    rf_boundary = pd.read_csv(RF_OUTPUT/'boundary_aware_metrics.csv')
    rows=[]
    for name,table in [('Random Forest',rf_boundary),('SVM',svm_boundary)]:
        for row in table[(table.split=='test') & (table.participant=='all') & table.tolerance_s.isin([0,1])].to_dict('records'):
            rows.append(dict(model=name,**row))
    comparison=pd.DataFrame(rows); comparison.to_csv(output/'comparison.csv',index=False)
    text=['# SVM comparison with saved Random Forest baseline','',
          'Same labeled coverage, 54 features, two-second overlapping windows, and 22/4/4 participant split. StandardScaler and SVM were fitted on training swimmers only. A fixed grid of 12 RBF SVM configurations and five decision thresholds was ranked on validation transition F1, then recall and precision; ties favor lower C and threshold. The selected model was not refitted on validation swimmers. Test results were calculated after selection.', '',
          f'Selected C={bundle["C"]:g}, gamma={bundle["gamma"]}, decision threshold={threshold:g}. Decision scores are margins, not probabilities; they are not directly comparable to the RF probability threshold.', '',
          '| Model | Boundary tolerance | Test accuracy | Transition precision | Transition recall | Transition F1 | Retained windows |',
          '|---|---|---:|---:|---:|---:|---:|']
    for r in comparison.itertuples():
        text.append(f'| {r.model} | {r.tolerance_s:g} s | {r.accuracy:.1%} | {r.transition_precision:.1%} | {r.transition_recall:.1%} | {r.transition_f1:.3f} | {r.windows} |')
    text += ['', 'One-second scores exclude boundary-adjacent centers; excluded windows are not credited as correct. Strict scores remain the primary baseline. These test swimmers were previously reviewed during RF development, so this is a comparison on an already inspected holdout, not a fresh independent final test. No SVM settings were selected using test scores. Only four swimmers are held out; overlapping windows are correlated.', '',
             'Reload without retraining from the repository root:', '', '```sh', '.venv/bin/python Code/compare_svm.py --mode evaluate', '```', '',
             'The saved pipeline contains both scaling and the SVM. Training refuses to overwrite it. Use --output-dir with a new directory for a separate experiment. RF files and original labels are unchanged.']
    (output/'README.md').write_text('\n'.join(text)+'\n')
    print(comparison[['model','tolerance_s','accuracy','transition_precision','transition_recall','transition_f1']].to_string(index=False))

if __name__ == '__main__':
    main()
