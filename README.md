# CluckSense

**Acoustic stress detection for poultry, built in MATLAB by Team PHAL.**

CluckSense listens to poultry sounds, scores how distressed the birds sound, and raises an alert when that score stays high. This repository holds the working software MVP: the sound-classification pipeline, the alert logic, and two dashboards.

> **Status:** software MVP complete. The system detects *respiratory-abnormal* sounds as a proof of concept for acoustic stress monitoring. It does not yet detect heat stress directly (see [Limitations](#limitations)).

---

## Table of contents

1. [Project overview](#project-overview)
2. [Objectives](#objectives)
3. [How it works](#how-it-works)
4. [Repository structure](#repository-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Implementation details](#implementation-details)
8. [Results](#results)
9. [Limitations](#limitations)
10. [Roadmap](#roadmap)
11. [Team and acknowledgements](#team-and-acknowledgements)
12. [License](#license)

---

## Project overview

Heat and respiratory problems in poultry houses can build up quickly, and a farmer cannot listen to every pen all day. Birds change how they sound when they are stressed, so sound is a cheap signal to monitor.

CluckSense turns audio into a **stress score from 0 to 100** and watches that score over time. A single noisy moment is ignored. An alert fires only when the score stays above a threshold for a sustained period, which keeps false alarms low.

## Objectives

1. Classify poultry audio into three classes: **Background_Noise**, **Healthy** and **Respiratory_Abnormal**.
2. Convert the classifier output into a continuous stress score that can be tracked over time.
3. Raise an alert only when high stress is *sustained*, not on single spikes.
4. Present the result on a simple dashboard that a farmer or reviewer can understand at a glance.
5. Keep the design light enough to move to a low-cost edge device (ESP32) in the next stage.

## How it works

```
Audio (.wav)
   -> YAMNet (pretrained audio network)         1024-number embedding per ~0.48 s frame
   -> average over a short window, standardize
   -> SVM classifier (ECOC)                     raw class scores
   -> softmax                                   scores between 0 and 1
   -> stress = Respiratory_Abnormal score x 100
   -> moving-average smoothing
   -> alert if stress > 70 for 30 readings in a row (about 14 s)
   -> dashboard
```

The heavy lifting is done by **YAMNet**, a network already trained on a large audio dataset. CluckSense uses it as a frozen feature extractor (transfer learning) and trains only a small SVM classifier on top.

## Repository structure

| File or folder | Purpose |
|---|---|
| `runCluckSense.m` | Main script: scores audio, computes stress, runs the alert logic, evaluates the model, opens the dashboard |
| `stressFromScores.m` | Converts class scores into a stress value for the target class |
| `stressMonitor.m` | Counts consecutive high readings and raises alerts |
| `plotStress.m` | Plots the stress curve with the threshold and alert markers |
| `loadAndEvaluate.m` | Reads the predictions CSV and reports accuracy, recall and precision |
| `evaluateModel.m` | Model evaluation helper |
| `CluckSenseDashboard.m` | MATLAB dashboard (curve, alert lamp, gauge, metrics, confusion matrix) |
| `clucksense_dashboard.html` | Browser dashboard for demos: replays a timeline, optional audio, can load a stress CSV exported from MATLAB |
| `Phase1_Frozen_YAMNet_SVM_Model.mat` | Saved model: `classifier`, `featureMean`, `featureStd`, `classOrder`, `embeddingLayer`, `embeddingDimension` |
| `Phase1_Frozen_YAMNet_SVM_Test_Predictions.csv` | True and predicted class for every test recording |
| `demofiles/` | Recordings used to build the demo timeline |

## Installation

### Requirements

- **MATLAB** R2024a or later (MATLAB Online works). The code uses `audioPretrainedNetwork`, which needs a recent release.
- **Audio Toolbox** (`yamnetPreprocess`, `audioPretrainedNetwork`)
- **Deep Learning Toolbox** (`dlarray`, `predict` on a `dlnetwork`)
- **Statistics and Machine Learning Toolbox** (`ClassificationECOC` prediction)
- **Audio Toolbox Model for YAMNet Network** support package. Install it from **Home > Add-Ons > Get Add-Ons**.
- A modern web browser for the HTML dashboard (no installation needed).

### Setup

1. Get the code:
   ```bash
   git clone <REPOSITORY-URL>
   ```
   or download the files into one folder (on MATLAB Online, upload them to MATLAB Drive).
2. Make sure the folder contains the `.m` files, the model `.mat`, the predictions `.csv` and a `demofiles/` folder with the recordings you want to use.
3. In MATLAB, make that folder your current folder, or run `addpath("<your folder>")`.
4. Check that the model and YAMNet load:
   ```matlab
   S   = load("Phase1_Frozen_YAMNet_SVM_Model.mat");
   net = audioPretrainedNetwork("yamnet");
   ```
5. Update the file paths at the top of `runCluckSense.m` (the model path, the `demoFiles` list and `predictionsFile`) so they point to your folder.

## Usage

### Run the full pipeline

```matlab
runCluckSense
```

This will:

1. score each file in `demoFiles` and join the embeddings into one timeline,
2. compute the stress curve and the alerts,
3. plot the stress curve with the 70% threshold,
4. print the model accuracy, recall and precision,
5. open the MATLAB dashboard.

The script has a flag, `useFakeScores`, that switches between a synthetic test curve (for checking the alert logic) and scoring real audio with the model.

### Use the browser dashboard

1. Export your stress curve from MATLAB:
   ```matlab
   writematrix(stress(:), "stress.csv");
   ```
2. Download `stress.csv`, open `clucksense_dashboard.html`, and click **Load stress CSV**.
3. Optionally click **Choose chicken audio** to play a recording alongside.
4. Click **Play demo**. For long timelines the page picks a window of about a minute around the first alert.

## Implementation details

### 1. Feature extraction

Each recording is converted to a spectrogram with `yamnetPreprocess` and passed through YAMNet. The `global_average_pooling2d` layer gives a 1024-number embedding for every ~0.48 s frame.

```matlab
spec = yamnetPreprocess(x, fs);
out  = predict(net, dlarray(spec, "SSCB"), Outputs=S.embeddingLayer);
emb  = reshape(extractdata(out), 1024, [])';      % frames x 1024
```

### 2. Classification

The classifier is an ECOC SVM (`ClassificationECOC`). It was trained on one standardized vector per recording, so embeddings are averaged over a window and standardized with the saved `featureMean` and `featureStd` before prediction.

```matlab
v = (mean(emb(i:i+winFrames-1, :), 1) - S.featureMean) ./ S.featureStd;
[~, negloss] = predict(S.classifier, v);
```

The model has no probability output (`ScoreTransform = 'none'`), so the raw scores are converted to a 0 to 1 scale with a softmax:

```matlab
e = exp(k * (double(negloss) - max(negloss)));
scores(i, :) = e / sum(e);
```

### 3. Stress and alerts

The stress value is the Respiratory_Abnormal score scaled to 0 to 100, smoothed with a moving average. `stressMonitor` counts consecutive readings above the threshold and fires one alert per sustained stretch.

| Parameter | Value | Meaning |
|---|---|---|
| `threshold` | 70 | Stress level counted as high |
| `streakLimit` | 30 readings | High readings in a row needed for an alert (about 14.4 s) |
| `winFrames` | 3 | Frames averaged per reading (about 1.5 s) |
| `k` | 5 | Softmax sharpness |
| smoothing | `movmean`, window 10 | Removes single-reading spikes |
| seconds per reading | 0.48 | One YAMNet frame hop |

These values are working settings chosen for the demo, not calibrated values.

### 4. Dashboards

- **MATLAB (`CluckSenseDashboard.m`)**: a `uifigure` with the stress curve and alert markers, an alert lamp, a gauge, a per-class metrics table and the confusion matrix. Metrics are read straight from the predictions CSV.
- **Browser (`clucksense_dashboard.html`)**: a single self-contained page with the risk card, persistence bar, stress curve, audio-level bars and a phone view that receives the alert message.

## Results

Evaluated on the held-out test split (55 recordings):

| Class | Recall | Precision |
|---|---|---|
| Background_Noise | 0.79 | 0.65 |
| Healthy | 0.86 | 0.90 |
| Respiratory_Abnormal | 0.84 | 0.94 |

**Overall accuracy: 83.6%**

Confusion matrix (rows = true class, columns = predicted):

| | Background_Noise | Healthy | Respiratory_Abnormal |
|---|---|---|---|
| **Background_Noise** | 11 | 2 | 1 |
| **Healthy** | 3 | 19 | 0 |
| **Respiratory_Abnormal** | 3 | 0 | 16 |

On the demo timeline (7 recordings joined, 713 readings), the alert fires during sustained abnormal stretches and ignores brief spikes.

## Limitations

- **Not heat stress yet.** The classes are Healthy, Respiratory_Abnormal and Background_Noise. Heat-stress-labelled audio is needed to detect heat stress directly.
- **The stress score is an index, not a probability.** The SVM has no probability output, and the 0 to 100 scale depends on the softmax setting. The threshold of 70 is a working value.
- **Windowed scoring is an approximation.** The model was trained on whole-recording vectors, so short-window scores are noisier than the test accuracy suggests.
- **The demo timeline is simulated.** It is built from separate recordings joined together. Some demo recordings were chosen by ear and are not all from the test split, so the timeline shows how the pipeline behaves and is not evidence of accuracy. Accuracy comes from the test split only.
- **Background_Noise precision is lower (0.65).** Other sounds are sometimes mistaken for noise.
- **Not real time.** The dashboards replay scores computed beforehand. Live microphone scoring is not implemented.

## Roadmap

1. Add a replay mode to the MATLAB dashboard that plays audio in step with the curve.
2. Run a lightweight model on an **ESP32** with real sensor streaming (ThingSpeak becomes useful at this stage).
3. Collect or find audio labelled for heat stress and retrain.
4. Calibrate the stress score and tune the threshold on real farm data.

## Team and acknowledgements

- **Team PHAL**: [ADD TEAM MEMBER NAMES AND ROLES]
- Built at FUT Minna, Nigeria.
- **YAMNet** is a pretrained audio-event network from Google, used here through MATLAB Audio Toolbox.
- Dataset: [ADD DATASET SOURCE AND LICENSE]

## License

[ADD LICENSE]
