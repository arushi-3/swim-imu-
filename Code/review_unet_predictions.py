#!/usr/bin/env python3
"""Review saved U-Net predictions against manual ranges and reference windows."""
from pathlib import Path
import html
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import validate_random_forest as baseline

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'Code/outputs/unet_comparison'


def main():
    samples=pd.read_csv(OUTPUT/'sample_predictions.csv',dtype={'participant':str})
    windows=pd.read_csv(OUTPUT/'all_window_predictions.csv',dtype={'participant':str})
    labels=pd.read_csv(baseline.LABELS/'transition_ranges.csv',dtype={'participant':str})
    boundary=pd.read_csv(OUTPUT/'boundary_annotated_predictions.csv',dtype={'participant':str})
    review=OUTPUT/'review_graphs'; review.mkdir(exist_ok=True)
    summaries=[]; errors=[]; items=[]
    for recording, frame in windows[windows.split=='test'].groupby('recording'):
        participant,filename=recording.split('/')
        point=samples[samples.recording==recording]
        raw=pd.read_csv(baseline.rf.DEFAULT_DATA_DIR/recording)
        raw['time']=(raw.timestamp-raw.timestamp.iloc[0])/1e9
        raw=raw[raw.time.between(point.time.min(),point.time.max())]
        ranges=labels[(labels.participant==participant)&(labels.file==filename)]
        missed=frame[(frame.label==0)&(frame.prediction==1)]
        false=frame[(frame.label==1)&(frame.prediction==0)]
        nearby=boundary[(boundary.recording==recording)&(boundary.label!=boundary.prediction)]
        boundary_errors=int(nearby.boundary_adjacent_1s.sum())
        summary=dict(participant=participant,recording=recording,total_windows=len(frame),
                     actual_transition_windows=int((frame.label==0).sum()),actual_clean_windows=int((frame.label==1).sum()),
                     correctly_detected_transition_windows=int(((frame.label==0)&(frame.prediction==0)).sum()),
                     correctly_detected_clean_windows=int(((frame.label==1)&(frame.prediction==1)).sum()),
                     missed_transition_windows=len(missed),clean_windows_flagged_transition=len(false),
                     errors_near_boundary_1s=boundary_errors,errors_away_from_boundary_1s=len(missed)+len(false)-boundary_errors,
                     accuracy=float((frame.label==frame.prediction).mean()))
        summaries.append(summary)
        wrong=nearby.copy(); wrong['error_type']=wrong.label.map({0:'missed_transition',1:'clean_flagged_transition'})
        errors.append(wrong)
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.09,
                         subplot_titles=['Sensor signals and your marked transition periods',
                                         'Two-second window labels and predictions (0 = transition, 1 = clean)',
                                         'Sample-level clean probability and misclassified windows'])
        for axis in ['GYRO_1','GYRO_2']:
            fig.add_trace(go.Scatter(x=raw.time,y=raw[axis],name=axis,mode='lines'),row=1,col=1)
        for r in ranges.itertuples():
            fig.add_vrect(x0=r.start_s,x1=r.end_s,fillcolor='orange',opacity=.22,line_width=0,row=1,col=1)
        fig.add_trace(go.Scatter(x=[None],y=[None],mode='markers',marker=dict(color='orange',symbol='square'),name='Your marked transition periods'),row=1,col=1)
        for column,name,symbol,color in [('label','Manual window label','circle','black'),('prediction','U-Net window prediction','x','purple')]:
            fig.add_trace(go.Scatter(x=frame.center_time,y=frame[column],name=name,mode='markers',
                                    marker=dict(symbol=symbol,color=color,size=7),customdata=frame[['start_time','end_time']].to_numpy(),
                                    hovertemplate='Window %{customdata[0]:.2f}–%{customdata[1]:.2f}s<br>Class %{y}<extra>%{fullData.name}</extra>'),row=2,col=1)
        fig.add_trace(go.Scatter(x=point.time,y=point.clean_probability,name='Sample P(clean)',mode='lines',line=dict(color='gray')),row=3,col=1)
        threshold=json.loads((OUTPUT/'model_metadata.json').read_text())['clean_threshold']
        fig.add_hline(y=threshold,line_dash='dash',annotation_text=f'Sample threshold {threshold:.2f}',row=3,col=1)
        for subset,name,color in [(missed,'Missed transition window','crimson'),(false,'Clean window flagged as transition','royalblue')]:
            for r in subset.itertuples():
                fig.add_vrect(x0=r.start_time,x1=r.end_time,fillcolor=color,opacity=.13,line_width=0,row=3,col=1)
            fig.add_trace(go.Scatter(x=subset.center_time,y=subset.mean_sample_clean_probability,name=name,mode='markers',
                                    marker=dict(color=color,size=9),customdata=subset[['start_time','end_time']].to_numpy(),
                                    hovertemplate='Window %{customdata[0]:.2f}–%{customdata[1]:.2f}s<br>Mean P(clean) %{y:.3f}<extra>%{fullData.name}</extra>'),row=3,col=1)
        fig.update_yaxes(range=[-.15,1.15],row=2,col=1)
        fig.update_yaxes(range=[-.05,1.05],row=3,col=1)
        fig.update_xaxes(title_text='Seconds from recording start',row=3,col=1)
        fig.update_layout(title=f'U-Net test swimmer {participant}: {filename}<br><sup>{len(missed)} missed transition windows | {len(false)} clean windows flagged | accuracy {summary["accuracy"]:.1%}</sup>',
                          height=1050,template='plotly_white',hovermode='x unified',legend=dict(orientation='h'),margin=dict(t=130))
        name=recording.replace('/','_').replace('.csv','_unet.html')
        fig.write_html(review/name,include_plotlyjs=True)
        items.append(f'<tr><td><a href="{html.escape(name)}">{participant}</a></td><td>{len(frame)}</td><td>{len(missed)}</td><td>{len(false)}</td><td>{boundary_errors}</td><td>{summary["accuracy"]:.1%}</td></tr>')
    pd.DataFrame(summaries).to_csv(review/'window_error_summary.csv',index=False)
    pd.concat(errors,ignore_index=True).to_csv(review/'misclassified_test_windows.csv',index=False)
    page='''<!doctype html><html><head><meta charset="utf-8"><title>U-Net error review</title><style>body{font:17px system-ui;max-width:1000px;margin:45px auto;padding:0 20px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #ddd}p{line-height:1.6}</style></head><body><h1>U-Net test error review</h1><p>Each window covers two seconds, with one-second spacing. Manual window labels and U-Net window predictions are clean only when at least 90% of their samples are clean. Overlapping windows can count the same event more than once.</p><table><thead><tr><th>Swimmer</th><th>Windows</th><th>Missed transitions</th><th>Clean flagged</th><th>Errors near boundary (1s)</th><th>Accuracy</th></tr></thead><tbody>'''+''.join(items)+'''</tbody></table><p><strong>Orange:</strong> your manually marked transition periods. <strong>Red:</strong> transition windows predicted clean. <strong>Blue:</strong> clean windows predicted transition. The middle panel compares window classes; the bottom panel shows sample probabilities and error spans.</p><p>The sample probability cutoff applies to samples, not mean window probability. Boundary proximity is descriptive: errors are still counted in the strict scores. No retraining or label changes were made.</p><p><a href="window_error_summary.csv">Window error summary CSV</a> · <a href="misclassified_test_windows.csv">Misclassified window details CSV</a></p></body></html>'''
    (review/'index.html').write_text(page)
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__': main()
