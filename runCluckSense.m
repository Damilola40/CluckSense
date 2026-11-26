clc;
clear;
close all;

classNames = ["Background_Noise" "Healthy" "Respiratory_Abnormal"];
targetClass = "Respiratory_Abnormal";
threshold = 70;  streakLimit = 30;  useFakeScores = false;

addpath(genpath(pwd))

predictionsFile = "results/Phase1_Frozen_YAMNet_SVM_Test_Predictions.csv";

if useFakeScores
    readings = [ ...
        42 41 43 40 44 42 45 43 41 44 ...   %  1-10   calm
        46 45 47 44 48 46 49 47 45 48 ...   % 11-20   calm
        50 52 49 51 53 50 54 52 51 55 ...   % 21-30   warming
        58 56 60 57 59 62 61 58 60 63 ...   % 31-40   rising
        66 64 68 65 67 70 69 66 68 71 ...   % 41-50   crossing 70 (71 = first high)
        74 72 76 73 75 78 77 74 76 79 ...   % 51-60   sustained high  (10 highs)
        82 80 84 81 83 86 85 82 84 87 ...   % 61-70   sustained high  (20 highs)
        88 86 89 87 88 85 86 88 87 89 ...   % 71-80   peak            (30 highs -> ALERT at #80)
        84 82 85 83 81 84 82 80 83 81 ...   % 81-90   sustained high  (40 highs total)
        78 75 72 70 68 65 62 58 55 50 ...   % 91-100  falling back
        ];
    s = readings(:) / 100;

    scores = zeros(numel(s), 3);

    % Third column: Respiratory_Abnormal
    scores(:, 3) = s;

    % First and second columns share the remaining probability
    scores(:, 1) = (1 - s) / 2;
    scores(:, 2) = (1 - s) / 2;

else
    S = load('/MATLAB Drive/CluckSense/models/Phase1_Frozen_YAMNet_SVM_Model.mat');
    net = audioPretrainedNetwork("yamnet");
    folder = "/MATLAB Drive/CluckSense/";
    demoFiles = [ ...
        "/MATLAB Drive/CluckSense/demofiles/16.wav" ...
        "/MATLAB Drive/CluckSense/demofiles/35.wav" ...
        "/MATLAB Drive/CluckSense/demofiles/48.wav" ...
        "/MATLAB Drive/CluckSense/demofiles/52.wav" ...
        "/MATLAB Drive/CluckSense/demofiles/64.wav" ...
        "/MATLAB Drive/CluckSense/demofiles/65.wav" ...
        "/MATLAB Drive/CluckSense/demofiles/80.wav" ];   % calm -> abnormal
    winFrames = 3;     % ~1.5 s windows, tune this
    k = 5;             % softmax sharpness, tune this

    frameCount = zeros(1, numel(demoFiles));    % before the loop
    allEmb = [];

    for n = 1:numel(demoFiles)
        [x, fs] = audioread(demoFiles(n));
        x = mean(x, 2);
        spec = yamnetPreprocess(x, fs);
        out = predict(net, dlarray(spec, "SSCB"), Outputs=S.embeddingLayer);
        embN = reshape(extractdata(out), 1024, [])';
        frameCount(n) = size(embN, 1);          % this file only
        allEmb = [allEmb; embN]; %#ok<AGROW>
    end

    nWin = size(allEmb, 1) - winFrames + 1;
    scores = zeros(nWin, 3);
    for i = 1:nWin
        v = (mean(allEmb(i:i+winFrames-1, :), 1) - S.featureMean) ./ S.featureStd;
        [~, nl] = predict(S.classifier, v);
        e = exp(k * (double(nl) - max(nl)));
        scores(i, :) = e / sum(e);
    end
end

% 4. Calculate stress
stress = stressFromScores(scores, classNames, targetClass);
stress = movmean(stress, 10);
b = [0 cumsum(frameCount)];
for n = 1:numel(demoFiles)
    idx = (b(n)+1):min(b(n+1), numel(stress));
    fprintf("%s: %.0f%% of readings above %d\n", demoFiles(n), 100*mean(stress(idx) > threshold), threshold);
end

% 5. Monitor stress and detect alerts
[alerts, alertIdx] = stressMonitor(stress', threshold, streakLimit);

above = stress(:)' > threshold;
d = diff([0 above 0]);
runs = find(d == -1) - find(d == 1);
fprintf("Readings: %d | max stress: %.1f | above threshold: %d | longest run: %d (need %d)\n", ...
    numel(stress), max(stress), sum(above), max([runs 0]), streakLimit);

writematrix(stress(:), "stress.csv");

% 6. Plot stress
plotStress(stress, alertIdx, threshold);

% 7. Display alerts
if isempty(alerts)
    disp("No alerts");
else
    for i = 1:numel(alerts)
        disp(alerts(i));
    end
end

% 8. Evaluate model
results = loadAndEvaluate(predictionsFile, classNames);

% 9. Run Dashboard
CluckSenseDashboard(stress, alertIdx, threshold, predictionsFile, classNames);
