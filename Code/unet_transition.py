#!/usr/bin/env python3
"""Small six-channel 1D U-Net. Train once; --mode evaluate reloads weights.
Eight-second inputs, sample-level labels, training-only normalization,
validation early stopping and threshold selection, held-out evaluation.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import validate_random_forest as baseline

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'Code/outputs/unet_comparison'
CHANNELS=['ACC_0','ACC_1','ACC_2','GYRO_0','GYRO_1','GYRO_2']
LENGTH=240; STRIDE=120; SEED=42


class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv1d(incoming,outgoing,5,padding=2),nn.ReLU(),
                                  nn.Conv1d(outgoing,outgoing,5,padding=2),nn.ReLU())
    def forward(self,x): return self.layers(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1=Block(6,8); self.enc2=Block(8,16); self.bottom=Block(16,32)
        self.pool=nn.MaxPool1d(2)
        self.up2=nn.ConvTranspose1d(32,16,2,stride=2); self.dec2=Block(32,16)
        self.up1=nn.ConvTranspose1d(16,8,2,stride=2); self.dec1=Block(16,8)
        self.out=nn.Conv1d(8,1,1)
    def forward(self,x):
        a=self.enc1(x); b=self.enc2(self.pool(a)); c=self.bottom(self.pool(b))
        d=self.dec2(torch.cat([self.up2(c),b],dim=1))
        return self.out(self.dec1(torch.cat([self.up1(d),a],dim=1))).squeeze(1)


def starts(n):
    if n<=LENGTH: return [0]
    return sorted(set(list(range(0,n-LENGTH+1,STRIDE))+[n-LENGTH]))


def load_sections():
    coverage=pd.read_csv(baseline.LABELS/'label_coverage.csv',dtype={'participant':str})
    labels=pd.read_csv(baseline.LABELS/'transition_ranges.csv',dtype={'participant':str})
    splits=pd.read_csv(baseline.LABELS/'participant_split.csv',dtype={'participant':str})
    mapping=dict(zip(splits.participant,splits.split)); sections=[]
    for (participant,filename), group in coverage.groupby(['participant','file'],sort=False):
        path=baseline.rf.DEFAULT_DATA_DIR/participant/filename
        raw=pd.read_csv(path); raw['time']=(raw.timestamp-raw.timestamp.iloc[0])/1e9
        fs=1/np.median(np.diff(raw.time))
        if abs(fs-30)>0.1: raise ValueError('Expected 30 Hz recordings')
        raw=baseline.rf.add_manual_labels(raw,list(labels[(labels.participant==participant)&(labels.file==filename)][['start_s','end_s']].itertuples(index=False,name=None)))
        for index,r in enumerate(group.itertuples()):
            data=raw[raw.time.between(r.labeled_start_s,r.labeled_end_s)]
            x=data[CHANNELS].to_numpy(dtype=np.float32)
            sections.append(dict(recording=f'{participant}/{filename}',participant=participant,split=mapping[participant],
                                 section=index,x=x,y=(1-data.clean_swimming.to_numpy()).astype(np.float32),time=data.time.to_numpy()))
    return sections


def chunks(sections,mean,std):
    xs=[]; ys=[]; masks=[]
    for section in sections:
        x=((section['x']-mean)/std).T
        for start in starts(len(section['y'])):
            count=min(LENGTH,len(section['y'])-start)
            sample=np.zeros((6,LENGTH),dtype=np.float32); target=np.zeros(LENGTH,dtype=np.float32)
            mask=np.zeros(LENGTH,dtype=np.float32)
            sample[:,:count]=x[:,start:start+count]; target[:count]=section['y'][start:start+count]; mask[:count]=1
            xs.append(sample); ys.append(target); masks.append(mask)
    return TensorDataset(torch.tensor(np.array(xs)),torch.tensor(np.array(ys)),torch.tensor(np.array(masks)))


def infer(model,section,mean,std):
    n=len(section['y']); sums=np.zeros(n); counts=np.zeros(n)
    model.eval()
    with torch.no_grad():
        for start in starts(n):
            count=min(LENGTH,n-start)
            x=np.zeros((6,LENGTH),dtype=np.float32)
            x[:,:count]=((section['x'][start:start+count]-mean)/std).T
            probability=torch.sigmoid(model(torch.from_numpy(x[None])))[0,:count].numpy()
            sums[start:start+count]+=probability; counts[start:start+count]+=1
    assert (counts>0).all()
    return sums/counts


def scores(y,pred):
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    return dict(observations=len(y),accuracy=float(accuracy_score(y,pred)),transition_precision=float(p[0]),
                transition_recall=float(r[0]),transition_f1=float(f[0]),clean_f1=float(f[1]),
                confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['train','evaluate'],default='train')
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--patience',type=int,default=12)
    args=parser.parse_args(); OUTPUT.mkdir(parents=True,exist_ok=True)
    artifact=OUTPUT/'unet.pt'
    if args.mode=='train' and artifact.exists(): raise ValueError('Model exists; use --mode evaluate')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    # Reuse the strict audit and reference window definitions.
    windows=baseline.build(baseline.rf.DEFAULT_DATA_DIR,OUTPUT)
    sections=load_sections(); training=[s for s in sections if s['split']=='train']; validation=[s for s in sections if s['split']=='validation']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [baseline.LABELS/name for name in ['transition_ranges.csv','label_coverage.csv','participant_split.csv']]}
    model=UNet()
    if args.mode=='train':
        unique=np.concatenate([s['x'] for s in training]); mean=unique.mean(axis=0); std=unique.std(axis=0); std=np.maximum(std,1e-6)
        targets=np.concatenate([s['y'] for s in training]); positive_weight=float((targets==0).sum()/(targets==1).sum())
        loader=DataLoader(chunks(training,mean,std),batch_size=16,shuffle=True)
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight),reduction='none')
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
        history=[]; best=float('inf'); stale=0; best_epoch=0
        for epoch in range(1,args.epochs+1):
            model.train(); loss_sum=0.; count_sum=0
            for x,y,mask in loader:
                optimizer.zero_grad(); losses=loss_fn(model(x),y)
                loss=(losses*mask).sum()/mask.sum(); loss.backward(); optimizer.step()
                loss_sum+=float((losses.detach()*mask).sum()); count_sum+=float(mask.sum())
            probs=np.concatenate([infer(model,s,mean,std) for s in validation]); truth=np.concatenate([s['y'] for s in validation])
            clipped=np.clip(probs,1e-7,1-1e-7)
            val_loss=float(np.mean(-positive_weight*truth*np.log(clipped)-(1-truth)*np.log(1-clipped)))
            history.append(dict(epoch=epoch,training_loss=loss_sum/count_sum,validation_loss=val_loss))
            if val_loss<best-1e-5:
                best=val_loss; best_epoch=epoch; stale=0
                torch.save(dict(state_dict=model.state_dict(),mean=mean.tolist(),std=std.tolist(),label_hashes=hashes,
                                channels=CHANNELS,input_samples=LENGTH,stride_samples=STRIDE,positive_weight=positive_weight,
                                best_epoch=epoch,seed=SEED,torch_version=str(torch.__version__)),artifact)
            else: stale+=1
            print(f'Epoch {epoch}: train {loss_sum/count_sum:.4f}, validation {val_loss:.4f}',flush=True)
            if stale>=args.patience: break
        pd.DataFrame(history).to_csv(OUTPUT/'training_history.csv',index=False)
        saved=torch.load(artifact,map_location='cpu',weights_only=True); model.load_state_dict(saved['state_dict'])
        validation_p=np.concatenate([1-infer(model,s,mean,std) for s in validation]); validation_y=np.concatenate([1-s['y'] for s in validation]).astype(int)
        searches=[]
        for cutoff in np.arange(.3,.801,.05):
            searches.append(dict(clean_threshold=float(round(cutoff,2)),**scores(validation_y,(validation_p>=cutoff).astype(int))))
        search=pd.DataFrame(searches).sort_values(['transition_f1','transition_recall','transition_precision','clean_threshold'],ascending=[False,False,False,True],kind='stable')
        search.to_csv(OUTPUT/'threshold_results.csv',index=False)
        saved['clean_threshold']=float(search.iloc[0].clean_threshold); torch.save(saved,artifact)
    else:
        saved=torch.load(artifact,map_location='cpu',weights_only=True)
        if saved['label_hashes']!=hashes or saved['channels']!=CHANNELS or saved['input_samples']!=LENGTH or saved['stride_samples']!=STRIDE: raise ValueError('Inputs changed from saved model')
        model.load_state_dict(saved['state_dict']); mean=np.array(saved['mean'],dtype=np.float32); std=np.array(saved['std'],dtype=np.float32)
    threshold=saved['clean_threshold']; frames=[]
    for s in sections:
        probability=1-infer(model,s,mean,std)
        frames.append(pd.DataFrame(dict(recording=s['recording'],participant=s['participant'],split=s['split'],time=s['time'],
                                       label=(1-s['y']).astype(int),clean_probability=probability,prediction=(probability>=threshold).astype(int))))
    samples=pd.concat(frames,ignore_index=True); samples.to_csv(OUTPUT/'sample_predictions.csv',index=False)
    sample_metrics={split:scores(f.label,f.prediction) for split,f in samples.groupby('split')}
    (OUTPUT/'sample_metrics.json').write_text(json.dumps(sample_metrics,indent=2))
    output_windows=windows.copy(); predicted=[]; probabilities=[]
    for r in windows.itertuples():
        segment=samples[(samples.recording==r.recording)&(samples.time>=r.start_time-1e-8)&(samples.time<=r.end_time+1e-8)]
        if segment.empty: raise ValueError('Uncovered reference window')
        assert int(segment.label.mean()>=.90)==r.label
        predicted.append(int(segment.prediction.mean()>=.90)); probabilities.append(float(segment.clean_probability.mean()))
    output_windows['prediction']=predicted; output_windows['mean_sample_clean_probability']=probabilities
    output_windows.to_csv(OUTPUT/'all_window_predictions.csv',index=False)
    window_metrics={split:scores(f.label,f.prediction) for split,f in output_windows.groupby('split')}
    (OUTPUT/'metrics.json').write_text(json.dumps(window_metrics,indent=2))
    pd.DataFrame([dict(participant=p,split=s,**scores(f.label,f.prediction)) for (p,s),f in output_windows.groupby(['participant','split'])]).to_csv(OUTPUT/'participant_metrics.csv',index=False)
    import report_boundary_aware as boundary
    boundary.OUTPUT=OUTPUT; boundary.main()
    (OUTPUT/'model_metadata.json').write_text(json.dumps({k:v for k,v in saved.items() if k!='state_dict'},indent=2))
    import review_unet_predictions
    review_unet_predictions.main()
    history=pd.read_csv(OUTPUT/'training_history.csv'); fig=go.Figure()
    for column in ['training_loss','validation_loss']: fig.add_trace(go.Scatter(x=history.epoch,y=history[column],name=column))
    fig.update_layout(title='U-Net training history',xaxis_title='Epoch',yaxis_title='Weighted binary cross-entropy',template='plotly_white'); fig.write_html(OUTPUT/'training_curve.html')
    comparison=[]
    for name,folder in [('RF',baseline.OUTPUT),('SVM',baseline.OUTPUT.parent/'svm_comparison'),('U-Net',OUTPUT)]:
        m=json.loads((folder/'metrics.json').read_text())['test']
        comparison.append(dict(model=name,**{key:m[key] for key in ['accuracy','transition_precision','transition_recall','transition_f1']}))
    pd.DataFrame(comparison).to_csv(OUTPUT/'comparison.csv',index=False)
    lines=['# Small 1D U-Net comparison','',
           'Six raw channels, training-only normalization, eight-second (240-sample) chunks, four-second stride, and a two-level encoder/decoder with skip connections (8/16/32 channels). Overlapping inference probabilities are averaged. Padding is masked from loss. Labels are transition=1 during learning and clean=1 in reports. Training uses Adam, weighted BCE, seed 42, CPU, and validation-loss early stopping; the saved epoch is selected before validation threshold tuning. No test tuning.', '',
           f'Best epoch: {saved["best_epoch"]}; clean sample threshold: {threshold:.2f}.', '',
           '| Model | Strict test window accuracy | Transition precision | Transition recall | Transition F1 |','|---|---:|---:|---:|---:|']
    for r in comparison: lines.append(f'| {r["model"]} | {r["accuracy"]:.1%} | {r["transition_precision"]:.1%} | {r["transition_recall"]:.1%} | {r["transition_f1"]:.3f} |')
    lines+=['','## Evaluation definitions','',
            'Sample metrics are saved separately. For window comparison, each existing two-second reference window is clean only if at least 90% of its predicted samples are clean, mirroring the manual window-label rule. Mean sample probability is descriptive and does not directly determine window class. RF/SVM predict windows directly; U-Net aggregation is an explicit bridge, not identical task training.', '',
            'One-second boundary-aware window scores use the same center-distance exclusion as before. This is offline inference: predictions use future as well as past context. Manual boundary uncertainty remains; sample precision does not establish accurate physical transition timing. Eight-second chunks and short padded sections can affect edge predictions.', '',
            'Only 30 swimmers and about 63 minutes of coverage were available. Validation and test swimmers have been reviewed in earlier experiments, so this is exploratory. Overlapping samples/windows are correlated. A fresh holdout or participant-level cross-validation is needed for stronger generalization claims.', '',
            'Saved weights, normalization, threshold, label hashes and settings are in unet.pt. Reload without fitting:', '', '```sh','.venv/bin/python Code/unet_transition.py --mode evaluate','```','',
            'Run training only once; the script refuses to overwrite an existing artifact. Sensor files and labels must remain available. Original RF/SVM files are unchanged.']
    lines += ['', '## Initial-run findings', '',
              'The U-Net is an initial baseline, not an improvement over RF. Consult participant_metrics.csv: most false alarms in this run came from swimmer 33. High training/validation scores and a poor result on one test swimmer suggest a generalization gap, whose cause is not yet established. Further development should use training/validation participants, not optimize against these test errors.', '',
              f'Sample-level test accuracy: {sample_metrics["test"]["accuracy"]:.1%}; sample transition F1: {sample_metrics["test"]["transition_f1"]:.3f}. Sample scores cannot be directly compared with RF/SVM window scores.', '',
              f'Training completed {len(history)} epochs; the saved network has 16,433 parameters. Eight seconds is the processing chunk length; local convolutions mean each prediction does not use the entire eight seconds.', '',
              'Dependency versions are recorded in Code/requirements_unet_environment.txt. Architecture, padding and inference checks are in Code/test_unet_transition.py.', '',
              'Implementation references: [PyTorch Conv1d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv1d.html) and [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html).']
    (OUTPUT/'README.md').write_text('\n'.join(lines)+'\n'); print(pd.DataFrame(comparison).to_string(index=False))

if __name__=='__main__': main()
