# Swimming IMU research: Random Forest and SVM explained

## What I am trying to do

I am building a model that distinguishes **clean freestyle swimming** from **transitions**, such as turns, pauses, and other sections I marked as non-clean swimming.

An IMU records movement through accelerometer and gyroscope signals. I manually inspected recordings and marked transition periods. Those labels give the models examples to learn from and a reference for checking their predictions.

This stage detects clean swimming versus transitions. It does **not** directly count strokes or distinguish all swimming styles.

## What is a window?

A window is a short section of a recording. My code uses **two-second windows**, moving forward **one second** each time:

| Window | Approximate time covered |
|---|---|
| 1 | 0–2 seconds |
| 2 | 1–3 seconds |
| 3 | 2–4 seconds |

The recordings contain approximately 30 sensor samples per second, so each window contains about 60 samples. Neighboring windows share half their data.

A window is not necessarily one stroke or one transition. A single transition can span several windows. The 508 test windows come from four swimmers, not 508 separate swimmers or independent experiments.

## How the code prepares the data

Both models use the same preparation:

1. Read the recordings and my manual transition labels.
2. Use only sections I explicitly marked as inspected. Uninspected data is not assumed to be clean.
3. Divide each inspected section into two-second windows.
4. Give each window a reference label: **clean if at least 90% of its samples are outside my transition ranges; transition otherwise**.
5. Calculate **54 features** that summarize movement in each window.

Features include averages, variation, maximum and minimum values, signal strength, and measures of peaks and repeated movement. Statistics use all accelerometer and gyroscope axes plus their magnitudes. Peak and frequency features use GYRO_1. Magnetometer signals are not inputs to these models.

The models receive these numerical summaries, not screenshots of graphs. Each window is assessed independently; the models do not explicitly look at the longer sequence around it. This can make an isolated stroke amid silence difficult to interpret.

## How I split the swimmers

I have labeled coverage from **31 recordings belonging to 30 swimmers**. There are 78 transition-range entries, including a near-duplicate pair for swimmer 6. Overlapping ranges are interpreted as their combined coverage, so the duplicate does not label samples twice.

| Group | Swimmers | Windows | Purpose |
|---|---:|---:|---|
| Training | 22 | 2,735 | Teach the model using labeled examples |
| Validation | 4 | 516 | Choose settings and decision cutoffs |
| Test | 4 | 508 | Measure performance on swimmers excluded from training and validation |

A swimmer appears in only one group. This matters because the goal is to work on different swimmers, rather than merely recognize recordings from people the model already learned from.

All labeled coverage is included in the experiment, but only the training group teaches the model. Training and scoring on all the same swimmers would not measure performance on unseen swimmers.

## Random Forest: many decision trees

A decision tree learns a sequence of questions about a window, such as whether movement variation or gyroscope strength is above a learned cutoff.

A **Random Forest combines many trees**. Each contributes an opinion, and the forest combines those opinions into a prediction.

My RF code uses:

- 100 trees.
- A maximum tree depth of 4, keeping the trees relatively simple.
- At least 3 training windows in each final tree leaf.
- Balanced class weighting, so class frequency does not alone determine its emphasis.
- A fixed random seed of 42 for reproducibility in the same environment.

It trains on the training swimmers. It then uses validation swimmers to select a clean-swimming probability cutoff from 0.50–0.80. The selected cutoff is **0.65**:

> If the clean-swimming probability is at least 0.65, predict clean swimming. Otherwise, predict transition.

## SVM: finding a dividing boundary

A **Support Vector Machine (SVM)** learns a boundary separating clean windows from transition windows based on their features.

Imagine plotting windows using two movement measurements: clean swimming might gather in one region, and transitions in another. SVM tries to separate those regions. My model uses 54 features, so the actual boundary is more complex.

I use an **RBF kernel**, which allows a curved boundary rather than requiring a straight separation.

My SVM code:

- Scales the features using averages and variation calculated from training swimmers only.
- Trains 12 combinations of SVM settings.
- Checks five decision cutoffs for each combination on validation swimmers.
- Selects the combination using transition F1, with recall and precision as tie-breakers.
- Keeps the selected model trained on the training group; it does not refit using validation swimmers.

The selected settings were **C = 1, gamma = 0.1, and decision cutoff = 0**. C controls the tradeoff between a simpler separation and fitting training examples closely. Gamma controls how locally each example influences the curved boundary.

The SVM output is a **decision score**, not a probability. Its numerical cutoff cannot be directly compared with RF’s 0.65 probability cutoff.

## Main differences

| Question | Random Forest | SVM |
|---|---|---|
| How does it learn? | Combines many decision trees | Finds a boundary between the classes |
| Does it need feature scaling? | Usually unnecessary | Yes; scaling is included in my pipeline |
| What does my code output? | A clean-swimming probability | A decision score |
| What was selected using validation data? | Probability cutoff | Model settings and decision cutoff |
| What is shared? | Same labels, features, windows, and swimmer split | Same labels, features, windows, and swimmer split |

## What the evaluation measures mean

**Accuracy:** What fraction of all windows were classified correctly?

**Transition precision:** When the model predicts transition, how often is it correct? Lower precision means more clean windows are incorrectly flagged as transitions.

**Transition recall:** Of all actual transition windows, how many does the model find? Lower recall means more transitions are missed.

**Transition F1:** A combined measure of precision and recall. It ranges from 0 to 1, with 1 being perfect:

```text
F1 = 2 × precision × recall / (precision + recall)
```

F1 is useful because high accuracy alone can hide poor transition detection when clean windows are more common.

## My strict test results

These results include every test window:

| Metric | Random Forest | SVM |
|---|---:|---:|
| Accuracy | **94.9%** | 92.9% |
| Transition precision | **91.6%** | 84.7% |
| Transition recall | 91.6% | **93.5%** |
| Transition F1 | **0.916** | 0.889 |

There were 154 transition windows and 354 clean windows in the test group.

| Outcome | Random Forest | SVM |
|---|---:|---:|
| Correctly detected transition windows | 141 | 144 |
| Missed transition windows | 13 | 10 |
| Clean windows incorrectly called transition | 13 | 26 |
| Correctly classified clean windows | 341 | 328 |

SVM found three more transition windows, but also falsely flagged 13 more clean windows. **RF achieved the better overall balance in this experiment.**

## Why I also report boundary-aware results

I placed transition boundaries manually, so their exact timing has some uncertainty. Also, a two-second window near a boundary can contain both transition and swimming.

My supplementary report excludes windows whose **centers are within one second of an internal manually labeled transition start or end**. It does not change the model, labels, or predictions. Excluded windows are not counted as correct. The start and end of the inspected coverage are not treated as internal transition boundaries.

One second is a fixed descriptive assumption—half the window length—not a measured amount of human error. The report also shows 0.5- and 2-second tolerances to show sensitivity. I retain the strict scores as the primary baseline.

| Metric away from boundaries | Random Forest | SVM |
|---|---:|---:|
| Retained test windows | 467 | 467 |
| Accuracy | **97.4%** | 94.6% |
| Transition precision | **90.9%** | 82.8% |
| Transition recall | 100% | 100% |
| Transition F1 | **0.952** | 0.906 |

These results show strong detection of transition interiors. They do not establish that predicted transition start/end times are accurate.

## What I can conclude—and the limits

RF is the stronger baseline on this split. SVM is slightly more sensitive to transitions, but produces more false alarms.

The scores measure agreement with my manual labels at the window level. They do not yet measure stroke-count accuracy, exact boundary timing, or performance across every possible swimming condition.

Only four swimmers were held out, neighboring windows overlap, and most inspected sections come from recording beginnings. These factors limit how broadly I can generalize the results.

The test recordings were reviewed during RF development. The boundary-aware report was added after that review, so it is supplementary. SVM settings were chosen using validation swimmers, but its test comparison uses the already inspected holdout. A fresh held-out group would provide a stronger final evaluation after further development.

## Training once and using the saved models

Both fitted models are saved. I do not hardcode their learned decisions or retrain every time I run evaluation.

- The RF artifact stores the forest, feature order, settings, and probability cutoff.
- The SVM artifact stores feature scaling, the SVM, feature order, and decision cutoff.
- Evaluation reloads these artifacts. Reloading was checked to reproduce predictions without fitting again.
- A new training run is appropriate when I intentionally change the training data, features, or settings.

## Where the work is saved

Paths below are relative to the Swimming IMU Research project folder.

| Item | Location |
|---|---|
| RF workflow | `Code/validate_random_forest.py` |
| RF model | `Code/outputs/rf_all_labels_validation/random_forest.joblib` |
| RF results and review graphs | `Code/outputs/rf_all_labels_validation/` |
| Boundary-aware reporting | `Code/report_boundary_aware.py` |
| SVM workflow | `Code/compare_svm.py` |
| SVM model and scaling | `Code/outputs/svm_comparison/svm.joblib` |
| SVM results and comparison | `Code/outputs/svm_comparison/` |

The RF baseline was committed on `research/rf-all-labels-validation`. SVM work continues on `research/svm-comparison`, which starts from that baseline.

## A short explanation I can say aloud

> I am using swimming movement-sensor data to separate clean freestyle from transitions such as turns and pauses. I manually labeled sections from 30 swimmers and divided them into overlapping two-second windows. Each window is summarized using 54 movement features. I compared Random Forest and SVM using the same training, validation, and test swimmers. Random Forest performed better overall, with 94.9% test accuracy and a transition F1 of 0.916. I also report results away from manually placed boundaries because exact labeling times are uncertain. Both models are saved, and my next steps are to evaluate timing and generalization more thoroughly.
