#!/usr/bin/env python3
"""Create held-out error review graphs from saved RF predictions, without fitting."""
from pathlib import Path
import html
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import TransitionClassifier_multi_file_130s as rf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'Code' / 'outputs' / 'rf_all_labels_validation'


def main():
    predictions = pd.read_csv(OUTPUT / 'all_window_predictions.csv', dtype={'participant': str})
    transitions = pd.read_csv(ROOT / 'Code/labels/transition_ranges.csv', dtype={'participant': str})
    coverage = pd.read_csv(ROOT / 'Code/labels/label_coverage.csv', dtype={'participant': str})
    review = OUTPUT / 'review_graphs'
    review.mkdir(exist_ok=True)
    links, errors = [], []
    for recording, windows in predictions[predictions.split == 'test'].groupby('recording'):
        participant, filename = recording.split('/')
        raw = pd.read_csv(rf.DEFAULT_DATA_DIR / recording)
        raw['time'] = (raw.timestamp - raw.timestamp.iloc[0]) / 1e9
        sections = coverage[(coverage.participant == participant) & (coverage.file == filename)]
        mask = pd.Series(False, index=raw.index)
        for section in sections.itertuples():
            mask |= raw.time.between(section.labeled_start_s, section.labeled_end_s)
        data = raw[mask]
        labels = transitions[(transitions.participant == participant) & (transitions.file == filename)]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=['Gyroscope signals and manual transition ranges',
                                            'Window labels and predictions (0 = transition, 1 = clean)',
                                            'Clean-swimming probability'])
        for axis in ['GYRO_1', 'GYRO_2']:
            fig.add_trace(go.Scatter(x=data.time, y=data[axis], name=axis, mode='lines'), row=1, col=1)
        for interval in labels.itertuples():
            fig.add_vrect(x0=interval.start_s, x1=interval.end_s, fillcolor='orange', opacity=.18,
                          line_width=0, row=1, col=1)
        for column, name, symbol in [('label', 'Manual window label', 'circle'), ('prediction', 'RF window prediction', 'x')]:
            fig.add_trace(go.Scatter(x=windows.center_time, y=windows[column], name=name,
                                    mode='markers', marker=dict(symbol=symbol, size=7),
                                    hovertemplate='Time %{x:.2f}s<br>Class %{y}<extra>%{fullData.name}</extra>'), row=2, col=1)
        wrong = windows[windows.label != windows.prediction].copy()
        for kind, subset, color in [('Missed transition', wrong[wrong.label == 0], 'crimson'),
                                    ('Clean flagged as transition', wrong[wrong.label == 1], 'royalblue')]:
            for window in subset.itertuples():
                fig.add_vrect(x0=window.start_time, x1=window.end_time, fillcolor=color,
                              opacity=.12, line_width=0, row=3, col=1)
            fig.add_trace(go.Scatter(x=subset.center_time, y=subset.clean_probability,
                                    name=kind, mode='markers', marker=dict(color=color,size=10),
                                    customdata=subset[['start_time','end_time']].to_numpy(),
                                    hovertemplate='Window %{customdata[0]:.2f}–%{customdata[1]:.2f}s<br>P(clean) %{y:.3f}<extra>%{fullData.name}</extra>'), row=3, col=1)
        fig.add_trace(go.Scatter(x=windows.center_time, y=windows.clean_probability,
                                name='P(clean)', mode='lines', line=dict(color='gray')), row=3, col=1)
        threshold = float(windows.clean_probability_threshold.iloc[0])
        fig.add_hline(y=threshold, line_dash='dash', annotation_text=f'Threshold {threshold:.2f}', row=3, col=1)
        fig.update_yaxes(range=[-.15,1.15], row=2,col=1)
        fig.update_yaxes(range=[-.05,1.05], row=3,col=1)
        fig.update_xaxes(title_text='Seconds from recording start', row=3,col=1)
        fig.update_layout(title=f'Test swimmer {participant}: {filename}<br><sup>{len(wrong)} errors / {len(windows)} windows; orange shading = manual transition ranges</sup>',
                          template='plotly_white', height=950, hovermode='x unified', legend=dict(orientation='h'), margin=dict(t=130))
        target = review / f'{participant}_{Path(filename).stem}_review.html'
        fig.write_html(target, include_plotlyjs=True)
        links.append((participant, target.name, len(wrong), len(windows)))
        errors.append(wrong)
    pd.concat(errors,ignore_index=True).to_csv(review / 'misclassified_test_windows.csv', index=False)
    items = ''.join(f'<li><a href="{html.escape(name)}">Swimmer {participant}</a>: {count} errors / {total} windows</li>' for participant,name,count,total in sorted(links,key=lambda x:-x[2]))
    (review / 'index.html').write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>RF test error review</title></head><body style="font:18px system-ui;max-width:850px;margin:50px auto"><h1>Random Forest test error review</h1><p>Saved model predictions on four held-out swimmers. Review windows against your manual labels; no model fitting or label changes were performed.</p><ul>{items}</ul><p>Orange shading shows manual transition intervals. The middle panel compares window labels and predictions. The bottom panel highlights missed transitions in red and clean windows flagged as transitions in blue. Overlapping windows can flag the same event more than once.</p><p>Use training and validation swimmers for subsequent tuning; retain these test results as the baseline.</p><a href="misclassified_test_windows.csv">Download misclassified windows</a></body></html>')
    print(review / 'index.html')
    print(pd.concat(errors).groupby(['participant','label']).size().to_string())

if __name__ == '__main__':
    main()
