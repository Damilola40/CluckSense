# CluckSense: What I Built and What I Learned

*Team PHAL | First real MATLAB project | Notes compiled 16 Nov 2026*

CluckSense listens to poultry audio, scores how stressed the birds sound, and raises an alert when the stress stays high. This document explains how every piece works, the errors we hit and why they happened, the results, and the limits. None of it is magic. Each stage is a small step you can check on its own.

---

## 1. The pipeline in one picture

```
Audio (.wav)
   |  yamnetPreprocess        -> turns sound into spectrogram patches
   v
YAMNet (pretrained network)
   |  predict(..., Outputs=embeddingLayer)
   v
Embeddings (frames x 1024)    -> a 1024-number "fingerprint" of each ~0.48 s of sound
   |  average over a window of frames, then standardize
   v
SVM classifier (ECOC)         -> raw scores for the 3 classes
   |  softmax
   v
Scores (readings x 3)         -> Background_Noise | Healthy | Respiratory_Abnormal
   |  stressFromScores        -> take the Respiratory_Abnormal column, scale to 0-100
   v
Stress curve                  -> smoothed with movmean
   |  stressMonitor
   v
Alert                         -> fires when 30 readings in a row are above 70
   |
   v
Dashboard                     -> curve, alert lamp, gauge, metrics, confusion matrix
```

**The key idea:** the heavy lifting (understanding sound) is done by YAMNet, a network Google already trained on millions of audio clips. We only trained a small, cheap SVM on top of its output. This is called **transfer learning** (here with a *frozen* feature extractor).

---

## 2. Files and what each one does

| File | Job |
|---|---|
| `Phase1_Frozen_YAMNet_SVM_Model.mat` | Your teammate's saved model: `classifier` (the SVM), `featureMean` and `featureStd` (for standardizing), `classOrder`, `embeddingLayer`, `embeddingDimension` |
| `Phase1_Frozen_YAMNet_SVM_Test_Predictions.csv` | True and predicted class for every test recording (55 rows) |
| `runCluckSense.m` | The main script: scores audio, computes stress, runs alerts, evaluates, opens the dashboard |
| `stressFromScores.m` | Turns the class scores into one stress number per reading |
| `stressMonitor.m` | Counts consecutive high readings and raises the alert |
| `plotStress.m` | Plots the stress curve with threshold and alert markers |
| `loadAndEvaluate.m` | Reads the predictions CSV and prints accuracy, recall and precision |
| `CluckSenseDashboard.m` | The dashboard (a `uifigure` built from code) |
| `demofiles/` | The recordings used to build the demo timeline |

---

## 3. Core concepts

**YAMNet.** A pretrained audio network that outputs, for every 0.96 s window of sound (stepping forward 0.48 s at a time), an embedding: 1024 numbers describing what the sound is like. Similar sounds get similar embeddings.

**Embedding.** A list of numbers that represents something (here, a slice of audio) so a simple classifier can work with it.

**Frozen.** We don't retrain YAMNet. We only use it to extract features.

**Standardizing.** Before the SVM sees an embedding, subtract `featureMean` and divide by `featureStd`. The SVM was trained on standardized data, so it must always be given standardized data. If you skip this, the predictions will be wrong, and nothing will crash to tell you.

**ECOC SVM (`ClassificationECOC`).** A classifier that handles 3+ classes by training several two-class SVMs and combining them.

**Why there are no probabilities.** The model's `ScoreTransform` is `'none'`, so `predict` returns raw scores (negative average loss), not probabilities. The highest score is the predicted class. The values are not between 0 and 1.

**Softmax.** A formula that turns any list of scores into numbers between 0 and 1 that add up to 1. We use it to make a 0-100 stress scale. A sharpness value `k` controls how extreme the result is. These outputs are a **stress index**, not calibrated probabilities.

**Windowing.** The SVM was trained on one vector per whole recording (the average of all its frames). To get a time series we score short sliding windows of frames instead. That is an approximation, and it makes the readings noisier.

**Smoothing (`movmean`).** Replaces each reading with the average of its neighbours, which removes single-reading spikes. Smoothing moved the longest high run from 38 to 54 readings, so alerts now fire inside sustained stretches.

**Threshold and streak.** Threshold = 70 means "high stress". Streak limit = 30 means "30 readings in a row above the threshold" before an alert fires. At 0.48 s per reading, that is about 14 s of sustained high stress. This stops one noisy reading from causing a false alarm.

**Confusion matrix.** Rows are the true class and columns are the predicted class. The diagonal is correct predictions.

**Precision and recall.**
- *Recall*: of all the real X, how many did we catch?
- *Precision*: of everything we called X, how many really were X?

---

## 4. The scoring code that works

```matlab
S   = load('/MATLAB Drive/CluckSense/Phase1_Frozen_YAMNet_SVM_Model.mat');
net = audioPretrainedNetwork("yamnet");

% one file -> embeddings
[x, fs] = audioread(file);
x    = mean(x, 2);                                   % force mono
spec = yamnetPreprocess(x, fs);
out  = predict(net, dlarray(spec, "SSCB"), Outputs=S.embeddingLayer);
emb  = reshape(extractdata(out), 1024, [])';         % frames x 1024

% one window -> class scores
v = (mean(emb(i:i+winFrames-1, :), 1) - S.featureMean) ./ S.featureStd;
[~, nl] = predict(S.classifier, v);                  % raw scores (negative loss)
e = exp(k * (double(nl) - max(nl)));                 % softmax, subtracting max for stability
scores(i, :) = e / sum(e);
```

Settings used: `winFrames = 3` (about 1.5 s per window), `k = 5`, smoothing `movmean(stress, 10)`.

---

## 5. Errors we hit, why they happened, and what they teach

| Error or symptom | Cause | Lesson |
|---|---|---|
| `whos -file "/MATLAB Drive/..."` said it could not find `/MATLAB.mat` | MATLAB's command syntax splits on spaces, so the space in "MATLAB Drive" cut the path | When a path has spaces, use function syntax: `whos('-file', 'path with spaces')` |
| `Unrecognized function 'yamnetEmbeddings'` | That function does not exist; I gave you the wrong name | Check that a function is real (`which`, `doc`) before building on it |
| `Incorrect number or types of inputs for function activations` | Newer YAMNet is a `dlnetwork`, and `activations` does not take it | For a `dlnetwork`, use `predict(net, dlarray(x,"SSCB"), Outputs=layerName)`. Check `class(net)` |
| `Dimensions of arrays being concatenated are not consistent` | A line break inside `[ ]` starts a new row, and the rows had different lengths | Continue a list across lines with `...`, or use `;` only when you really want rows |
| `audioread` could not find the file | The folder was added to a name that already had the full path, and a placeholder file name was still in the list | Decide once whether a variable holds a full path or a file name. Check with `isfile(...)` |
| NaN in the per-file diagnostic | I used `size(allEmb,1)` inside the loop, which is a running total, so the file boundaries were wrong | Store each file's own frame count separately, then append |
| `emb` not a valid variable | The loop had no variable called `emb`; I wrote it from memory of the earlier script | Variables must exist before you use them. Read the code you actually have |
| Alert time printed as `7050 s` | An older `stressMonitor` assumed 10 s per reading | Print a sanity check (does 705 readings x 0.48 s = 338 s make sense?) |
| `%d` with a non-whole number | `elapsedS = i * 0.48` is rarely an integer | Use `%.0f` for numbers that may have decimals |
| `useFakeScores` flag read backwards | We flipped the `if` with `~` to switch branches | Set the flag to what it means and keep the `if` natural |
| Dashboard showed NaN and zeros | `readtable` split the CSV on backslashes in the Windows file paths | When data looks wrong, inspect it (`head(T)`, `T.Properties.VariableNames`) before assuming the code is wrong. We fixed it by reading the CSV as text lines and splitting on commas |
| Warning: column headers were modified | `readtable` renames headers that are not valid MATLAB names | Use `"VariableNamingRule","preserve"`, and never assume header names. Look at them |

---

## 6. Debugging habits that paid off

1. **Check sizes at every stage.** `size(emb)` returning 1024 columns told us the YAMNet step was right before we touched the SVM.
2. **Reproduce a known answer.** `4.wav` is truly Background_Noise but your teammate's model predicts Respiratory_Abnormal. Our pipeline gave Respiratory_Abnormal too. Matching a mistake is the strongest proof that two pipelines are identical.
3. **Print a one-line diagnostic.** `Readings | max stress | above threshold | longest run` showed us why the alert was not firing (scattered highs, not sustained ones).
4. **Don't tune a threshold to force a result.** Lowering `streakLimit` until the alert fires would hide a real problem. Fix the data or the smoothing first.
5. **Keep fake data for testing logic, real data for testing the model.** The fake-score run proved the alert logic. The real run proved the model connection.

---

## 7. Results

**Model, on the test split (55 recordings)**

- Overall accuracy: **83.6%**
- Macro F1: about 82.7%

| Class | Recall | Precision |
|---|---|---|
| Background_Noise | 0.79 | 0.65 |
| Healthy | 0.86 | 0.90 |
| Respiratory_Abnormal | 0.84 | 0.94 |

Confusion matrix (rows = true, columns = predicted):

| | Background_Noise | Healthy | Respiratory_Abnormal |
|---|---|---|---|
| **Background_Noise** | 11 | 2 | 1 |
| **Healthy** | 3 | 19 | 0 |
| **Respiratory_Abnormal** | 3 | 0 | 16 |

**Demo timeline (7 recordings stitched together, 713 readings at 0.48 s each)**

- Max stress 97.2, with 219 readings above the threshold
- Longest run above threshold: 54 (the alert needs 30)
- Alerts at readings 188, 227, 354 and 703 (about 90 s, 109 s, 170 s and 337 s)
- Three files read high (35, 52, 80, with 54 to 74% of readings above 70), four read near zero

---

## 8. Honest limitations (say these before anyone asks)

1. **The classes are Healthy vs Respiratory_Abnormal vs Background_Noise, not heat stress.** CluckSense is a proof of concept for acoustic stress detection. True heat-stress labels need different data.
2. **The stress score is an index, not a probability.** The SVM has no probability output, and the 0-100 scale depends on the choice of `k`. The threshold of 70 and `k = 5` are working values, not calibrated ones.
3. **Windowed scoring is an approximation.** The model was trained on whole-recording vectors, so short-window scores are noisier than the 83.6% suggests.
4. **The demo timeline is simulated.** It is stitched from separate recordings, so "stress rising over time" is staged. The demo files were picked by ear, and five of the seven (35, 48, 52, 64, 65) do not appear in the test split, so the model may have trained on them. The timeline demonstrates the pipeline. It is not evidence of accuracy. Accuracy comes from the test split only.
5. **Background_Noise precision is 0.65.** Other sounds are sometimes mistaken for noise. The target class is stronger (0.94 precision, 0.84 recall).
6. **File numbers repeat across classes.** For example `80.wav` exists as both a Healthy and a Respiratory_Abnormal test file, so a file name alone does not identify its class. Keep the class folder or add a prefix.
7. **Not real time.** The dashboard replays scores that were computed beforehand. Live microphone scoring is a separate build.

---

## 9. Status and what is next

**Done**
- Real model connected: YAMNet embeddings, standardization, SVM, stress, alert
- Evaluation reproduces the 83.6% result
- Static dashboard written (curve, alert lamp, gauge, metrics table, confusion matrix). Its metrics panels were fixed after the CSV-reading error, so confirm they now show numbers
- Team informed. The pitch deck is the submission format, and a teammate is handling it

**Next**
1. **Replay demo button** (planned for 17 Nov): play a chicken recording while the stress curve builds in step with the audio, ending with the threshold crossing and the red alert lamp. Call it a "demo replay", not live detection. Use test-split recordings that the model classifies correctly, and aim for a 60 to 90 s demo.
2. Screen-record the dashboard for the deck, so timing or audio problems on the day cannot break it.
3. Hardware stage: the CNN for the ESP32 and real sensor streaming. ThingSpeak becomes useful then, because a device can send live readings.
4. If better data turns up, retrain toward true heat-stress labels.

---

## 10. Quick glossary

| Term | Meaning |
|---|---|
| Embedding | A list of numbers that represents a piece of data |
| Transfer learning | Reusing a model trained on a big task as the starting point for a small one |
| Frozen | Not retrained |
| Frame | One YAMNet step, about 0.48 s of audio |
| Softmax | Turns scores into 0 to 1 values that sum to 1 |
| `dlnetwork` | MATLAB's newer deep learning network type; use `predict` with `Outputs` on it |
| `movmean` | Moving average, used here to smooth the stress curve |
| Streak | The number of consecutive readings above the threshold |
| Macro F1 | The average F1 score across classes, giving each class equal weight |
| `uifigure` | A MATLAB figure that can hold app components (tables, lamps, gauges) |
