clc;
clear;
close all;

classNames = ["normal" "stressed" "noise"];
trueLabels      = categorical(["normal" "stressed" "stressed" "noise" "normal"], classNames);
predictedLabels = categorical(["normal" "stressed" "normal"   "noise" "normal"], classNames);

results = evaluateModel(trueLabels, predictedLabels, classNames);