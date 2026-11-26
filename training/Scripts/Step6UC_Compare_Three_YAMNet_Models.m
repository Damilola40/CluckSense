
%% STEP 6U-C: COMPARATIVE VALIDATION OF THREE MODELS
% Project: CluckSense Smart Poultry
%
% Models:
% 1. Frozen YAMNet embeddings + SVM
% 2. Frozen-head three-class YAMNet
% 3. Selectively fine-tuned three-class YAMNet
%
% Evaluation:
% Same 50 validation recordings for all models.
% Recording-level predictions.
%
% The 55 test recordings are not accessed.

clear;
clc;

fprintf('\n');
fprintf('====================================================\n');
fprintf(' STEP 6U-C: THREE-MODEL VALIDATION COMPARISON\n');
fprintf('====================================================\n\n');

%% 1. PROJECT PATHS

projectRoot = pwd;
resultsFolder = fullfile(projectRoot, 'Results');

svmPath = fullfile(resultsFolder, ...
    'Frozen_YAMNet_Baseline_Model.mat');

embeddingPath = fullfile(projectRoot, ...
    'Features', 'YAMNet_Embeddings', ...
    'Frozen_YAMNet_Embeddings.mat');

frozenHeadPath = fullfile(resultsFolder, ...
    'YAMNet_FrozenHead_Best_Checkpoint.mat');

fineTunedPath = fullfile(resultsFolder, ...
    'YAMNet_Selective_FineTuned_Best_Checkpoint.mat');

inspectionPath = fullfile(resultsFolder, ...
    'FineTuning_Data_Inspection.mat');

outputMatPath = fullfile(resultsFolder, ...
    'Step6UC_Three_Model_Validation_Comparison.mat');

outputSummaryPath = fullfile(resultsFolder, ...
    'Step6UC_Model_Comparison_Summary.csv');

outputMetricsPath = fullfile(resultsFolder, ...
    'Step6UC_Per_Class_Metrics.csv');

%% 2. VERIFY INPUT FILES

requiredFiles = { ...
    svmPath, ...
    embeddingPath, ...
    frozenHeadPath, ...
    fineTunedPath, ...
    inspectionPath};

for i = 1:numel(requiredFiles)

    if ~isfile(requiredFiles{i})
        error('Required file missing: %s', requiredFiles{i});
    end

end

fprintf('All required model and dataset files found.\n');

%% 3. LOAD SAVED MODELS AND DATA

fprintf('\nLoading models and validation data...\n');

S = load(svmPath);
E = load(embeddingPath);
H = load(frozenHeadPath);
F = load(fineTunedPath);
D = load(inspectionPath);

svmModel = S.baselineModel;
frozenHeadNet = H.bestNet;
fineTunedNet = F.bestNet;

T = D.T;
embeddingTable = E.embeddingTable;
embeddings = E.embeddings;

poultryClasses = string(F.poultryClasses(:));

if ~isequal(string(H.poultryClasses(:)), poultryClasses)
    error('The frozen-head and fine-tuned class orders differ.');
end

if height(embeddingTable) ~= size(embeddings, 1)
    error('Embedding table and embedding matrix row counts differ.');
end

if size(embeddings, 2) ~= 1024
    error('Expected 1024-dimensional YAMNet embeddings.');
end

%% 4. SELECT VALIDATION RECORDINGS

validationMask = string(T.DatasetSplit) == "Validation";
validationTable = T(validationMask, :);

if height(validationTable) ~= 50
    error('Expected 50 validation recordings; found %d.', ...
        height(validationTable));
end

numberRecordings = height(validationTable);
numberClasses = numel(poultryClasses);

actualLabels = string(validationTable.ClassLabel);

fprintf('Validation recordings: %d\n', numberRecordings);
fprintf('Number of classes: %d\n', numberClasses);

fprintf('\nValidation class counts:\n');

for c = 1:numberClasses
    fprintf('%s: %d\n', poultryClasses(c), ...
        sum(actualLabels == poultryClasses(c)));
end

%% 5. PREPARE EMBEDDING VALIDATION DATA FOR SVM

fprintf('\nPreparing validation embeddings for SVM...\n');

embeddingFileNames = string(embeddingTable.FileName);
embeddingSplits = string(embeddingTable.DatasetSplit);

validationEmbeddingRows = zeros(numberRecordings, 1);

for i = 1:numberRecordings

    currentFile = string(validationTable.FileName(i));

    matchingRows = find( ...
        embeddingFileNames == currentFile & ...
        embeddingSplits == "Validation");

    if numel(matchingRows) ~= 1
        error(['Expected exactly one validation embedding for: ' ...
            char(currentFile)]);
    end

    validationEmbeddingRows(i) = matchingRows;

end

XValidation = double(embeddings(validationEmbeddingRows, :));

if any(~isfinite(XValidation), 'all')
    error('Non-finite values found in validation embeddings.');
end

%% 6. MODEL 1: FROZEN YAMNET EMBEDDINGS + SVM

fprintf('\n');
fprintf('-----------------------------------------------\n');
fprintf('MODEL 1: FROZEN YAMNET EMBEDDINGS + SVM\n');
fprintf('-----------------------------------------------\n');

featureMean = S.featureMean;
featureStd = S.featureStd;

if any(featureStd == 0)
    error('Zero feature standard deviation found.');
end

XValidationStandardized = ...
    (XValidation - featureMean) ./ featureStd;

svmPredicted = predict(svmModel, XValidationStandardized);

svmPredicted = string(svmPredicted(:));

fprintf('SVM validation predictions generated.\n');

%% 7. MODEL 2: FROZEN-HEAD YAMNET

fprintf('\n');
fprintf('-----------------------------------------------\n');
fprintf('MODEL 2: FROZEN-HEAD THREE-CLASS YAMNET\n');
fprintf('-----------------------------------------------\n');

targetSampleRate = 16000;
minimumDuration = 1.5;

[frozenHeadPredicted, frozenHeadScores] = ...
    evaluateYAMNetRecordings( ...
        frozenHeadNet, validationTable, ...
        poultryClasses, projectRoot, ...
        targetSampleRate, minimumDuration);

fprintf('Frozen-head validation predictions generated.\n');

%% 8. MODEL 3: SELECTIVELY FINE-TUNED YAMNET

fprintf('\n');
fprintf('-----------------------------------------------\n');
fprintf('MODEL 3: SELECTIVELY FINE-TUNED YAMNET\n');
fprintf('-----------------------------------------------\n');

[fineTunedPredicted, fineTunedScores] = ...
    evaluateYAMNetRecordings( ...
        fineTunedNet, validationTable, ...
        poultryClasses, projectRoot, ...
        targetSampleRate, minimumDuration);

fprintf('Fine-tuned validation predictions generated.\n');

%% 9. VERIFY PREDICTION COUNTS AND CLASS LABELS

predictionSets = { ...
    svmPredicted, ...
    frozenHeadPredicted, ...
    fineTunedPredicted};

modelNames = [ ...
    "Frozen YAMNet + SVM"; ...
    "Frozen-head YAMNet"; ...
    "Selective fine-tuned YAMNet"];

for m = 1:numel(predictionSets)

    predictions = predictionSets{m};

    if numel(predictions) ~= numberRecordings
        error('Model %d returned an incorrect prediction count.', m);
    end

    if any(~ismember(predictions, poultryClasses))
        error('Model %d returned an unknown class label.', m);
    end

end

%% 10. CALCULATE CONFUSION MATRICES AND CLASS METRICS

fprintf('\n');
fprintf('====================================================\n');
fprintf(' CALCULATING COMPARATIVE PERFORMANCE METRICS\n');
fprintf('====================================================\n');

allConfusionMatrices = cell(3, 1);
allMetricTables = cell(3, 1);

overallAccuracy = zeros(3, 1);
macroF1 = zeros(3, 1);
macroPrecision = zeros(3, 1);
macroRecall = zeros(3, 1);

for m = 1:3

    predictions = predictionSets{m};

    [cm, metricTable, accuracyValue, ...
        precisionValue, recallValue, f1Value] = ...
        calculateClassificationMetrics( ...
            actualLabels, predictions, poultryClasses);

    allConfusionMatrices{m} = cm;
    allMetricTables{m} = metricTable;

    overallAccuracy(m) = accuracyValue;
    macroPrecision(m) = mean(precisionValue);
    macroRecall(m) = mean(recallValue);
    macroF1(m) = mean(f1Value);

    fprintf('\nMODEL: %s\n', modelNames(m));

    fprintf('Accuracy: %.2f%%\n', 100 * accuracyValue);
    fprintf('Macro Precision: %.2f%%\n', ...
        100 * macroPrecision(m));
    fprintf('Macro Recall: %.2f%%\n', ...
        100 * macroRecall(m));
    fprintf('Macro F1-score: %.2f%%\n', 100 * macroF1(m));

    fprintf('\nConfusion matrix:\n');
    fprintf('Rows = Actual classes; Columns = Predicted classes\n');

    disp(array2table(cm, ...
        'VariableNames', cellstr(poultryClasses), ...
        'RowNames', cellstr(poultryClasses)));

    fprintf('Per-class metrics:\n');
    disp(metricTable);

end

%% 11. BUILD MODEL COMPARISON SUMMARY

comparisonTable = table( ...
    modelNames, ...
    overallAccuracy * 100, ...
    macroPrecision * 100, ...
    macroRecall * 100, ...
    macroF1 * 100, ...
    'VariableNames', { ...
    'Model', ...
    'AccuracyPercent', ...
    'MacroPrecisionPercent', ...
    'MacroRecallPercent', ...
    'MacroF1Percent'});

fprintf('\n');
fprintf('====================================================\n');
fprintf(' FINAL MODEL COMPARISON\n');
fprintf('====================================================\n');

disp(comparisonTable);

%% 12. BUILD COMBINED PER-CLASS METRICS TABLE

combinedMetrics = table();

for m = 1:3

    currentMetrics = allMetricTables{m};

    currentMetrics.Model = ...
        repmat(modelNames(m), height(currentMetrics), 1);

    combinedMetrics = [combinedMetrics; currentMetrics];

end

combinedMetrics = movevars( ...
    combinedMetrics, 'Model', 'Before', 1);

%% 13. CREATE RECORDING-LEVEL PREDICTION TABLE

predictionTable = table( ...
    string(validationTable.FileName), ...
    actualLabels, ...
    svmPredicted, ...
    frozenHeadPredicted, ...
    fineTunedPredicted, ...
    'VariableNames', { ...
    'FileName', ...
    'ActualClass', ...
    'SVM_Prediction', ...
    'FrozenHead_Prediction', ...
    'FineTuned_Prediction'});

predictionTable.SVM_Correct = ...
    predictionTable.ActualClass == predictionTable.SVM_Prediction;

predictionTable.FrozenHead_Correct = ...
    predictionTable.ActualClass == predictionTable.FrozenHead_Prediction;

predictionTable.FineTuned_Correct = ...
    predictionTable.ActualClass == predictionTable.FineTuned_Prediction;

%% 14. IDENTIFY RESPIRATORY-ABNORMAL PERFORMANCE

respiratoryClass = "Respiratory_Abnormal";

fprintf('\n');
fprintf('RESPIRATORY-ABNORMAL DETECTION\n');

for m = 1:3

    predictions = predictionSets{m};

    actualPositive = actualLabels == respiratoryClass;
    predictedPositive = predictions == respiratoryClass;

    TP = sum(actualPositive & predictedPositive);
    FP = sum(~actualPositive & predictedPositive);
    FN = sum(actualPositive & ~predictedPositive);

    respiratoryPrecision = TP / max(TP + FP, 1);
    respiratoryRecall = TP / max(TP + FN, 1);

    respiratoryF1 = ...
        2 * respiratoryPrecision * respiratoryRecall / ...
        max(respiratoryPrecision + respiratoryRecall, eps);

    fprintf('\n%s\n', modelNames(m));
    fprintf('TP: %d\n', TP);
    fprintf('FP: %d\n', FP);
    fprintf('FN: %d\n', FN);
    fprintf('Precision: %.2f%%\n', ...
        100 * respiratoryPrecision);
    fprintf('Recall: %.2f%%\n', ...
        100 * respiratoryRecall);
    fprintf('F1-score: %.2f%%\n', ...
        100 * respiratoryF1);

end

%% 15. IDENTIFY RECORDINGS WHERE MODELS DISAGREE

predictionTable.ModelAgreement = ...
    (predictionTable.SVM_Prediction == ...
     predictionTable.FrozenHead_Prediction) & ...
    (predictionTable.FrozenHead_Prediction == ...
     predictionTable.FineTuned_Prediction);

disagreementTable = predictionTable( ...
    ~predictionTable.ModelAgreement, :);

fprintf('\nRecordings with model disagreement: %d of %d\n', ...
    height(disagreementTable), numberRecordings);

%% 16. SAVE RESULTS

writetable(comparisonTable, outputSummaryPath);
writetable(combinedMetrics, outputMetricsPath);

save(outputMatPath, ...
    'comparisonTable', ...
    'combinedMetrics', ...
    'predictionTable', ...
    'disagreementTable', ...
    'allConfusionMatrices', ...
    'allMetricTables', ...
    'overallAccuracy', ...
    'macroPrecision', ...
    'macroRecall', ...
    'macroF1', ...
    'frozenHeadScores', ...
    'fineTunedScores', ...
    'poultryClasses', ...
    'modelNames');

fprintf('\n');
fprintf('====================================================\n');
fprintf(' STEP 6U-C COMPLETED SUCCESSFULLY\n');
fprintf('====================================================\n');

fprintf('Comparison MAT file:\n%s\n', outputMatPath);

fprintf('\nComparison summary CSV:\n%s\n', outputSummaryPath);

fprintf('\nPer-class metrics CSV:\n%s\n', outputMetricsPath);

fprintf('\nAll models evaluated on the same 50 validation recordings.\n');
fprintf('The 55 test recordings were not accessed.\n');

%% LOCAL FUNCTION 1: EVALUATE YAMNET AT RECORDING LEVEL

function [predictedLabels, averageScores] = ...
    evaluateYAMNetRecordings( ...
        net, validationTable, poultryClasses, ...
        projectRoot, targetSampleRate, minimumDuration)

    numberRecordings = height(validationTable);
    numberClasses = numel(poultryClasses);

    predictedLabels = strings(numberRecordings, 1);
    averageScores = zeros(numberRecordings, numberClasses);

    for r = 1:numberRecordings

        audioFile = char(string(validationTable.FileName(r)));

        if ~isfile(audioFile)
            error('Audio file not found: %s', audioFile);
        end

        [audio, originalSampleRate] = audioread(audioFile);

        if isempty(audio)
            error('Empty audio recording: %s', audioFile);
        end

        if size(audio, 2) > 1
            audio = mean(audio, 2);
        end

        audio = single(audio);

        if originalSampleRate ~= targetSampleRate
            audio = resample( ...
                audio, targetSampleRate, originalSampleRate);
        end

        minimumSamples = round( ...
            minimumDuration * targetSampleRate);

        if numel(audio) < minimumSamples
            audio(end+1:minimumSamples, 1) = 0;
        end

        spectrograms = yamnetPreprocess( ...
            audio, targetSampleRate);

        spectrograms = single(spectrograms);

        if size(spectrograms, 1) ~= 96 || ...
                size(spectrograms, 2) ~= 64 || ...
                size(spectrograms, 3) ~= 1

            error('Unexpected YAMNet input dimensions: %s', audioFile);

        end

        numberWindows = size(spectrograms, 4);
        totalScores = zeros(numberClasses, 1);

        windowBatchSize = 64;

        for firstWindow = 1:windowBatchSize:numberWindows

            lastWindow = min( ...
                firstWindow + windowBatchSize - 1, ...
                numberWindows);

            indices = firstWindow:lastWindow;

            batch = spectrograms(:, :, :, indices);

            dlX = dlarray(single(batch), 'SSCB');

            scores = predict(net, dlX);

            scoreValues = double(extractdata(scores));
            scoreValues = reshape(scoreValues, numberClasses, []);

            if size(scoreValues, 2) ~= numel(indices)
                error('Unexpected YAMNet output dimensions.');
            end

            totalScores = totalScores + sum(scoreValues, 2);

        end

        recordingScores = totalScores / numberWindows;

        averageScores(r, :) = recordingScores(:)';

        [~, predictedIndex] = max(recordingScores);

        predictedLabels(r) = poultryClasses(predictedIndex);

        if mod(r, 10) == 0 || r == numberRecordings
            fprintf('Processed %d/%d recordings.\n', ...
                r, numberRecordings);
        end

    end

end

%% LOCAL FUNCTION 2: CLASSIFICATION METRICS

function [cm, metricTable, accuracy, ...
    precision, recall, f1] = ...
    calculateClassificationMetrics( ...
        actualLabels, predictedLabels, poultryClasses)

    numberClasses = numel(poultryClasses);

    cm = zeros(numberClasses, numberClasses);

    for i = 1:numel(actualLabels)

        actualIndex = find( ...
            poultryClasses == actualLabels(i), 1);

        predictedIndex = find( ...
            poultryClasses == predictedLabels(i), 1);

        if isempty(actualIndex) || isempty(predictedIndex)
            error('Unknown class found during metric calculation.');
        end

        cm(actualIndex, predictedIndex) = ...
            cm(actualIndex, predictedIndex) + 1;

    end

    TP = diag(cm);
    FP = sum(cm, 1)' - TP;
    FN = sum(cm, 2) - TP;

    precision = TP ./ max(TP + FP, 1);
    recall = TP ./ max(TP + FN, 1);

    f1 = 2 .* precision .* recall ./ ...
        max(precision + recall, eps);

    accuracy = sum(TP) / sum(cm(:));

    metricTable = table( ...
        poultryClasses, TP, FP, FN, ...
        precision * 100, ...
        recall * 100, ...
        f1 * 100, ...
        'VariableNames', { ...
        'Class', 'TP', 'FP', 'FN', ...
        'PrecisionPercent', ...
        'RecallPercent', ...
        'F1Percent'});

end
