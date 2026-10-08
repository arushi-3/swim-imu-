#!/usr/bin/env python3
"""Validation-only ablation of temporal sensor features for RF and SVM.

Every candidate retains all 48 statistical features. Compares no temporal
features, six individual axes, and all six. Test swimmers are not scored.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import validate_random_forest as v
import compare_all_sensor_features as all_axes

OUTPUT = v.OUTPUT.parent / 'sensor_axis_validation'


def select_threshold(frame, values, thresholds):
    rows=[]
    for threshold in thresholds:
        candidate=frame.copy(); candidate['prediction']=(values >= threshold).astype(int)
        rows.append(dict(threshold=float(threshold),**v.metrics(candidate)))
    return pd.DataFrame(rows).sort_values(['transition_f1','transition_recall','transition_precision','threshold'],ascending=[False,False,False,True],kind='stable')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'ranking.csv').exists():
        raise ValueError('Results already exist; preserve this experiment rather than overwrite it')
    v.rf.extract_features=all_axes.extract_all_axes
    features=v.build(v.rf.DEFAULT_DATA_DIR,OUTPUT)
    # Full feature extraction is shared; fitting, scaling and ranking only
    # access train/validation. No test predictions or scores are generated.
    train=features[features.split=='train']; valid=features[features.split=='validation']
    columns=v.rf.get_feature_columns(features)
    statistical=[c for c in columns if not any(c.endswith('_'+name) for name in all_axes.TEMPORAL)]
    assert len(statistical)==48
    variants={'statistics_only':[],**{axis:[axis] for axis in v.rf.ACC_COLUMNS+v.rf.GYRO_COLUMNS},'all_six':v.rf.ACC_COLUMNS+v.rf.GYRO_COLUMNS}
    results=[]; searches=[]; participants=[]
    for name,axes in variants.items():
        selected=statistical+[f'{axis}_{feature}' for axis in axes for feature in all_axes.TEMPORAL]
        for kind in ['RF','SVM']:
            print(f'Fitting {kind}: {name}, {len(selected)} features',flush=True)
            if kind=='RF':
                model=v.rf.train_classifier(train,selected)
                values=v.rf.get_clean_probabilities(model,valid,selected)
                ranked=select_threshold(valid,values,v.rf.CLEAN_PROBABILITY_THRESHOLDS)
                best=ranked.iloc[0]; configuration={}
            else:
                candidates=[]; models={}
                for c in [0.1,1.0,10.0,100.0]:
                    for gamma in ['scale',0.001,0.1]:
                        pipeline=make_pipeline(StandardScaler(),SVC(C=c,gamma=gamma,kernel='rbf',class_weight='balanced'))
                        pipeline.fit(train[selected],train.label)
                        models[(c,str(gamma))]=pipeline
                        search=select_threshold(valid,pipeline.decision_function(valid[selected]),[-1.0,-0.5,0.0,0.5,1.0])
                        search['C']=c; search['gamma']=str(gamma)
                        candidates.append(search)
                ranked=pd.concat(candidates).sort_values(['transition_f1','transition_recall','transition_precision','C','threshold'],ascending=[False,False,False,True,True],kind='stable')
                best=ranked.iloc[0]; model=models[(best.C,best.gamma)]
                values=model.decision_function(valid[selected]); configuration={'C':float(best.C),'gamma':best.gamma}
            threshold=float(best.threshold)
            result=dict(model=kind,variant=name,feature_count=len(selected),threshold=threshold,**configuration,
                        **{key:best[key] for key in ['accuracy','transition_precision','transition_recall','transition_f1']})
            results.append(result)
            ranked['model']=kind; ranked['variant']=name; searches.append(ranked)
            predictions=valid.copy(); predictions['prediction']=(values>=threshold).astype(int)
            predictions.to_csv(OUTPUT/f'{kind}_{name}_validation_predictions.csv',index=False)
            for participant,frame in predictions.groupby('participant'):
                participants.append(dict(model=kind,variant=name,participant=participant,**v.metrics(frame)))
            joblib.dump(dict(model=model,feature_columns=selected,threshold=threshold,axes=axes,
                             score_type='probability' if kind=='RF' else 'decision_margin',configuration=configuration),
                        OUTPUT/f'{kind}_{name}.joblib')
    ranking=pd.DataFrame(results).sort_values(['model','transition_f1','transition_recall','transition_precision'],ascending=[True,False,False,False],kind='stable')
    ranking.to_csv(OUTPUT/'ranking.csv',index=False)
    pd.concat(searches).to_csv(OUTPUT/'validation_search.csv',index=False)
    pd.DataFrame(participants).to_csv(OUTPUT/'participant_metrics.csv',index=False)
    lines=['# Which axes add useful peak/frequency features?','',
           'Validation-only ablation. Every model keeps the same 48 statistical features from all six sensor channels and magnitudes. Only the source of six peak/frequency features changes. Statistics-only uses 48 features, individual-axis variants use 54, and all-six uses 84. This does not compare raw channels alone or prove a physical axis is universally best.', '',
           'The same 22 training and four validation swimmers, covered sections and overlapping two-second windows are used throughout. RF settings are fixed. SVM uses the same training-only scaling and fixed C/gamma search for each variant. Thresholds and settings are selected on strict validation transition F1. Test swimmers are not scored or used to rank variants.', '',
           '| Model | Temporal feature axes | Accuracy | Transition precision | Transition recall | Transition F1 |',
           '|---|---|---:|---:|---:|---:|']
    for r in ranking.itertuples():
        lines.append(f'| {r.model} | {r.variant} | {r.accuracy:.1%} | {r.transition_precision:.1%} | {r.transition_recall:.1%} | {r.transition_f1:.3f} |')
    lines += ['', '## Interpretation','']
    for kind in ['RF','SVM']:
        group=ranking[ranking.model==kind]; winner=group.iloc[0]
        ties=group[np.isclose(group.transition_f1,winner.transition_f1,atol=1e-12,rtol=0)]
        lines.append(f'{kind}: highest validation transition F1 is {winner.transition_f1:.3f}; tied variants: {", ".join(ties.variant)}.')
    lines += ['', 'Small differences may reflect these four validation swimmers rather than a robust advantage. Scores are used for model selection and are optimistic estimates of generalization. One validation swimmer has only transition windows. Overlapping windows are correlated. Confirm a chosen feature set using participant-level cross-validation within training data or a fresh holdout before claiming a universally best axis.', '',
              'Per-swimmer scores, search results, validation predictions and fitted candidate models are saved alongside this report. Existing RF/SVM baselines and labels are preserved. No test accuracy is reported for this axis selection experiment.']
    (OUTPUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(ranking.to_string(index=False),flush=True)

if __name__=='__main__':
    main()
