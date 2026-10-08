# Small 1D U-Net comparison

Six raw channels, training-only normalization, eight-second (240-sample) chunks, four-second stride, and a two-level encoder/decoder with skip connections (8/16/32 channels). Overlapping inference probabilities are averaged. Padding is masked from loss. Labels are transition=1 during learning and clean=1 in reports. Training uses Adam, weighted BCE, seed 42, CPU, and validation-loss early stopping; the saved epoch is selected before validation threshold tuning. No test tuning.

Best epoch: 39; clean sample threshold: 0.65.

| Model | Strict test window accuracy | Transition precision | Transition recall | Transition F1 |
|---|---:|---:|---:|---:|
| RF | 94.9% | 91.6% | 91.6% | 0.916 |
| SVM | 92.9% | 84.7% | 93.5% | 0.889 |
| U-Net | 82.5% | 63.7% | 98.1% | 0.772 |

## Evaluation definitions

Sample metrics are saved separately. For window comparison, each existing two-second reference window is clean only if at least 90% of its predicted samples are clean, mirroring the manual window-label rule. Mean sample probability is descriptive and does not directly determine window class. RF/SVM predict windows directly; U-Net aggregation is an explicit bridge, not identical task training.

One-second boundary-aware window scores use the same center-distance exclusion as before. This is offline inference: predictions use future as well as past context. Manual boundary uncertainty remains; sample precision does not establish accurate physical transition timing. Eight-second chunks and short padded sections can affect edge predictions.

Only 30 swimmers and about 63 minutes of coverage were available. Validation and test swimmers have been reviewed in earlier experiments, so this is exploratory. Overlapping samples/windows are correlated. A fresh holdout or participant-level cross-validation is needed for stronger generalization claims.

Saved weights, normalization, threshold, label hashes and settings are in unet.pt. Reload without fitting:

```sh
.venv/bin/python Code/unet_transition.py --mode evaluate
```

Run training only once; the script refuses to overwrite an existing artifact. Sensor files and labels must remain available. Original RF/SVM files are unchanged.

## Initial-run findings

The U-Net is a baseline, not an improvement over RF. Most test false alarms came from swimmer 33: all 76 clean reference windows were called transitions. Test window accuracy by swimmer was 98.4% (14), 93.4% (28), 40.3% (33), and 98.4% (38). The strong training/validation results and poor results for swimmer 33 suggest a generalization gap; the cause has not been established. Keep the RF baseline and use training/validation participants for any further development.

Sample-level test accuracy was 87.5%, with transition F1 0.815. These sample scores cannot be compared directly to RF/SVM window scores.

Training stopped after 51 epochs, retaining epoch 39. The network has 16,433 learned parameters. Architecture and normalization are fixed for this run. The eight-second chunk is the processing length; local convolutions mean an individual prediction does not use the entire eight seconds.

Checks passed for output alignment, gradients, padding masks, and complete overlapping inference coverage. Reloading reproduced every sample probability without training. The dependency snapshot is in `Code/requirements_unet_environment.txt`.

Implementation references: [PyTorch Conv1d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv1d.html) and [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html).
