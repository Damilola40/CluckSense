
%% STEP 6T: FULL SELECTIVE FINE-TUNING OF YAMNET
% Project: CluckSense Smart Poultry
%
% Starting point:
%   Results\YAMNet_FrozenHead_Best_Checkpoint.mat
%
% Training data:
%   240 training recordings
%
% Validation data:
%   50 validation recordings
%
% Test data:
%   55 recordings - NOT USED IN THIS SCRIPT
%
% Trainable layers:
%   conv2d_13
%   dense
%
% All other learnable parameters remain frozen.
% Batch Normalization state remains unchanged.

clear;
clc;

fprintf('\n');
fprintf('====================================================\n');
fprintf(' STEP 6T: SELECTIVE YAMNET FINE-TUNING\n');
fprintf('====================================================\n\n');

%% 1. PROJECT PATHS

projectRoot = pwd;

checkpointPath = fullfile(projectRoot, ...
    'Results', 'YAMNet_FrozenHead_Best_Checkpoint.mat');

inspectionPath = fullfile(projectRoot, ...
    'Results', 'FineTuning_Data_Inspection.mat');

bestCheckpointPath = fullfile(projectRoot, ...
    'Results', 'YAMNet_Selective_FineTuned_Best_Checkpoint.mat');

finalModelPath = fullfile(projectRoot, ...
    'Results', 'YAMNet_Selective_FineTuned_Final.mat');

lastEpochPath = fullfile(projectRoot, ...
    'Results', 'YAMNet_Selective_FineTuned_LastEpoch.mat');

historyCsvPath = fullfile(projectRoot, ...
    'Results', 'YAMNet_Selective_FineTuning_History.csv');

if ~isfolder(fullfile(projectRoot, 'Results'))
    error('The Results folder was not found.');
end

if ~isfile(checkpointPath)
    error('Frozen-head best checkpoint was not found: %s', checkpointPath);
end

if ~isfile(inspectionPath)
    error('Fine-tuning inspection file was not found: %s', inspectionPath);
end

%% 2. LOAD THE BEST FROZEN-HEAD CHECKPOINT

fprintf('Loading the best frozen-head checkpoint...\n');

C = load(checkpointPath);

requiredCheckpointFields = { ...
    'bestNet', ...
    'bestEpoch', ...
    'bestValidationAccuracy', ...
    'poultryClasses'};

for k = 1:numel(requiredCheckpointFields)
    if ~isfield(C, requiredCheckpointFields{k})
        error('Missing checkpoint field: %s', ...
            requiredCheckpointFields{k});
    end
end

fineTuneNet = C.bestNet;

if ~isa(fineTuneNet, 'dlnetwork')
    error('The checkpoint network is not a dlnetwork.');
end

fprintf('Starting checkpoint epoch: %d\n', C.bestEpoch);
fprintf('Starting validation accuracy: %.2f%%\n', ...
    100 * C.bestValidationAccuracy);

%% 3. LOAD THE VERIFIED DATA MANIFEST

fprintf('\nLoading the verified dataset manifest...\n');

D = load(inspectionPath);

if ~isfield(D, 'T')
    error('The inspection file does not contain the expected T table.');
end

T = D.T;

requiredColumns = {'FileName', 'ClassLabel', 'DatasetSplit'};

for k = 1:numel(requiredColumns)
    if ~ismember(requiredColumns{k}, T.Properties.VariableNames)
        error('Manifest is missing column: %s', requiredColumns{k});
    end
end

poultryClasses = string(C.poultryClasses(:));

if numel(poultryClasses) ~= 3
    error('Exactly three poultry classes are required.');
end

fprintf('Class order:\n');
disp(poultryClasses);

trainingMask = string(T.DatasetSplit) == "Training";
validationMask = string(T.DatasetSplit) == "Validation";

trainTable = T(trainingMask, :);
validationTable = T(validationMask, :);

if height(trainTable) ~= 240
    error('Expected 240 training recordings; found %d.', ...
        height(trainTable));
end

if height(validationTable) ~= 50
    error('Expected 50 validation recordings; found %d.', ...
        height(validationTable));
end

% Do not create or load a test table in this training script.

fprintf('Training recordings: %d\n', height(trainTable));
fprintf('Validation recordings: %d\n', height(validationTable));
fprintf('Test recordings: reserved and untouched.\n');

%% 4. TRAINING CONFIGURATION

targetSampleRate = 16000;
minimumDuration = 1.5;

maxEpochs = 30;
miniBatchSize = 16;
patience = 7;

globalLearnRate = 0.0001;

selectedLayers = ["conv2d_13", "dense"];

convLearnRateFactor = 0.1;
denseLearnRateFactor = 1.0;

rng(42, 'twister');

fprintf('\nTRAINING CONFIGURATION\n');
fprintf('Maximum epochs: %d\n', maxEpochs);
fprintf('Mini-batch size: %d\n', miniBatchSize);
fprintf('Global learning rate: %.6f\n', globalLearnRate);
fprintf('Conv2d learning-rate factor: %.2f\n', ...
    convLearnRateFactor);
fprintf('Dense learning-rate factor: %.2f\n', ...
    denseLearnRateFactor);
fprintf('Early-stopping patience: %d epochs\n', patience);

%% 5. SET LEARNING-RATE FACTORS

fprintf('\nConfiguring selective parameter training...\n');

learnableTable = fineTuneNet.Learnables;

for i = 1:height(learnableTable)

    layerName = string(learnableTable.Layer(i));
    parameterName = string(learnableTable.Parameter(i));

    % Freeze every learnable parameter first.
    fineTuneNet = setLearnRateFactor( ...
        fineTuneNet, char(layerName), char(parameterName), 0);

end

% Enable only conv2d_13 and dense.
for i = 1:height(learnableTable)

    layerName = string(learnableTable.Layer(i));
    parameterName = string(learnableTable.Parameter(i));

    if layerName == "conv2d_13"
        factor = convLearnRateFactor;

    elseif layerName == "dense"
        factor = denseLearnRateFactor;

    else
        continue;
    end

    fineTuneNet = setLearnRateFactor( ...
        fineTuneNet, char(layerName), char(parameterName), factor);

end

% Verify the number of trainable parameter groups.
learnableTable = fineTuneNet.Learnables;

trainableMask = false(height(learnableTable), 1);

for i = 1:height(learnableTable)

    layerName = string(learnableTable.Layer(i));

    if any(layerName == selectedLayers)
        trainableMask(i) = true;
    end

end

numberTrainable = sum(trainableMask);
numberFrozen = sum(~trainableMask);

fprintf('Trainable parameter groups: %d\n', numberTrainable);
fprintf('Frozen parameter groups: %d\n', numberFrozen);

if numberTrainable ~= 4 || numberFrozen ~= 106
    error(['Unexpected parameter configuration. ' ...
        'Expected 4 trainable and 106 frozen groups.']);
end

%% 6. SNAPSHOT FROZEN PARAMETERS AND BATCH NORMALIZATION STATE

fprintf('\nSaving parameter and Batch Normalization snapshots...\n');

startingLearnables = fineTuneNet.Learnables;
startingState = fineTuneNet.State;

frozenParameterSnapshot = cell(numberFrozen, 1);
frozenIndices = find(~trainableMask);

for k = 1:numberFrozen
    frozenParameterSnapshot{k} = ...
        extractdata(startingLearnables.Value{frozenIndices(k)});
end

stateSnapshot = cell(height(startingState), 1);

for k = 1:height(startingState)
    stateSnapshot{k} = extractdataIfNeeded(startingState.Value{k});
end

fprintf('Frozen parameter groups recorded: %d\n', numberFrozen);
fprintf('Batch Normalization state entries recorded: %d\n', ...
    height(startingState));

%% 7. PREPARE TRAINING LABELS

trainLabels = string(trainTable.ClassLabel);

trainClassIndices = zeros(height(trainTable), 1);

for i = 1:height(trainTable)

    classIndex = find(poultryClasses == trainLabels(i), 1);

    if isempty(classIndex)
        error('Unknown training class: %s', trainLabels(i));
    end

    trainClassIndices(i) = classIndex;

end

fprintf('\nTRAINING CLASS COUNTS\n');

for k = 1:numel(poultryClasses)
    fprintf('%s: %d recordings\n', poultryClasses(k), ...
        sum(trainClassIndices == k));
end

%% 8. INITIALIZE TRAINING VARIABLES

trainingLossHistory = nan(maxEpochs, 1);
validationAccuracyHistory = nan(maxEpochs, 1);

bestValidationAccuracy = -inf;
bestEpoch = 0;
epochsWithoutImprovement = 0;

bestNet = [];

averageGrad = [];
averageSqGrad = [];
iteration = 0;

%% 9. BEGIN FULL SELECTIVE FINE-TUNING

fprintf('\n');
fprintf('====================================================\n');
fprintf(' STARTING FULL FINE-TUNING\n');
fprintf('====================================================\n');

for epoch = 1:maxEpochs

    fprintf('\n-----------------------------------------------\n');
    fprintf('EPOCH %d OF %d\n', epoch, maxEpochs);
    fprintf('-----------------------------------------------\n');

    % Randomize recording order.
    shuffledIndices = randperm(height(trainTable));

    % One random YAMNet window per recording this epoch.
    epochInputs = zeros(96, 64, 1, height(trainTable), 'single');
    epochTargets = zeros(3, height(trainTable), 'single');

    fprintf('Preparing one random window per training recording...\n');

    for n = 1:height(trainTable)

        rowIndex = shuffledIndices(n);

        audioFile = resolveAudioPath( ...
            trainTable.FileName(rowIndex), projectRoot);

        spectrograms = loadRecordingSpectrograms( ...
            audioFile, targetSampleRate, minimumDuration);

        numberWindows = size(spectrograms, 4);

        selectedWindow = randi(numberWindows);

        epochInputs(:, :, :, n) = ...
            single(spectrograms(:, :, :, selectedWindow));

        epochTargets(trainClassIndices(rowIndex), n) = 1;

        if mod(n, 40) == 0
            fprintf('Prepared %d of %d recordings.\n', ...
                n, height(trainTable));
        end

    end

    %% MINI-BATCH TRAINING

    numberBatches = ceil(height(trainTable) / miniBatchSize);

    epochLossTotal = 0;
    epochObservationCount = 0;

    for batchNumber = 1:numberBatches

        firstIndex = (batchNumber - 1) * miniBatchSize + 1;

        lastIndex = min( ...
            batchNumber * miniBatchSize, height(trainTable));

        batchIndices = firstIndex:lastIndex;

        XBatch = epochInputs(:, :, :, batchIndices);
        TBatch = epochTargets(:, batchIndices);

        dlX = dlarray(single(XBatch), 'SSCB');
        dlTargets = dlarray(single(TBatch), 'CB');

        % Calculate loss and gradients.
        [loss, gradients] = dlfeval( ...
            @modelGradients, fineTuneNet, dlX, dlTargets);

        % Check the loss before updating.
        lossValue = double(extractdata(loss));

        if ~isfinite(lossValue)
            error('Non-finite training loss at epoch %d, batch %d.', ...
                epoch, batchNumber);
        end

        % Explicitly zero gradients for all frozen parameters.
        for g = 1:height(gradients)

            layerName = string(gradients.Layer(g));

            if ~any(layerName == selectedLayers)

                gradients.Value{g} = zeros( ...
                    size(gradients.Value{g}), ...
                    'like', gradients.Value{g});

            end

        end

        % Perform the Adam update.
        iteration = iteration + 1;

        [fineTuneNet, averageGrad, averageSqGrad] = adamupdate( ...
            fineTuneNet, gradients, ...
            averageGrad, averageSqGrad, ...
            iteration, globalLearnRate);

        epochLossTotal = epochLossTotal + ...
            lossValue * numel(batchIndices);

        epochObservationCount = epochObservationCount + ...
            numel(batchIndices);

        if mod(batchNumber, 5) == 0 || batchNumber == numberBatches
            fprintf('Batch %d/%d | Loss: %.6f\n', ...
                batchNumber, numberBatches, lossValue);
        end

    end

    meanEpochLoss = epochLossTotal / epochObservationCount;
    trainingLossHistory(epoch) = meanEpochLoss;

    %% VERIFY FROZEN PARAMETERS AND BATCH NORMALIZATION STATE

    currentLearnables = fineTuneNet.Learnables;

    for k = 1:numberFrozen

        currentValue = extractdata( ...
            currentLearnables.Value{frozenIndices(k)});

        if ~isequal(currentValue, frozenParameterSnapshot{k})
            error('A frozen parameter changed during epoch %d.', epoch);
        end

    end

    currentState = fineTuneNet.State;

    if height(currentState) ~= numel(stateSnapshot)
        error('Batch Normalization state structure changed.');
    end

    for k = 1:height(currentState)

        currentValue = extractdataIfNeeded(currentState.Value{k});

        if ~isequal(currentValue, stateSnapshot{k})
            error('Batch Normalization state changed at epoch %d.', epoch);
        end

    end

    fprintf('\nFrozen parameter verification: PASSED\n');
    fprintf('Batch Normalization state verification: PASSED\n');

    %% VALIDATION AT RECORDING LEVEL

    fprintf('\nEvaluating validation recordings...\n');

    [validationAccuracy, validationPredictions] = ...
        evaluateRecordingLevel( ...
            fineTuneNet, validationTable, ...
            poultryClasses, projectRoot, ...
            targetSampleRate, minimumDuration);

    validationAccuracyHistory(epoch) = validationAccuracy;

    fprintf('\nEPOCH %d SUMMARY\n', epoch);
    fprintf('Mean training loss: %.6f\n', meanEpochLoss);
    fprintf('Validation accuracy: %.2f%% (%d/%d)\n', ...
        100 * validationAccuracy, ...
        round(validationAccuracy * height(validationTable)), ...
        height(validationTable));

    %% SAVE BEST CHECKPOINT

    if validationAccuracy > bestValidationAccuracy

        bestValidationAccuracy = validationAccuracy;
        bestEpoch = epoch;
        bestNet = fineTuneNet;
        epochsWithoutImprovement = 0;

        save(bestCheckpointPath, ...
            'bestNet', ...
            'bestEpoch', ...
            'bestValidationAccuracy', ...
            'poultryClasses', ...
            'trainingLossHistory', ...
            'validationAccuracyHistory', ...
            'validationPredictions', ...
            'globalLearnRate', ...
            'miniBatchSize', ...
            'selectedLayers');

        fprintf('\nNEW BEST MODEL SAVED.\n');
        fprintf('Best epoch: %d\n', bestEpoch);
        fprintf('Best validation accuracy: %.2f%%\n', ...
            100 * bestValidationAccuracy);

    else

        epochsWithoutImprovement = epochsWithoutImprovement + 1;

        fprintf('\nNo validation improvement.\n');
        fprintf('Early-stopping counter: %d/%d\n', ...
            epochsWithoutImprovement, patience);

    end

    %% SAVE TRAINING HISTORY AFTER EACH EPOCH

    historyTable = table( ...
        (1:epoch)', ...
        trainingLossHistory(1:epoch), ...
        validationAccuracyHistory(1:epoch) * 100, ...
        'VariableNames', { ...
        'Epoch', 'TrainingLoss', 'ValidationAccuracyPercent'});

    writetable(historyTable, historyCsvPath);

    %% EARLY STOPPING

    if epochsWithoutImprovement >= patience

        fprintf('\nEARLY STOPPING ACTIVATED.\n');
        fprintf('No validation improvement for %d epochs.\n', patience);
        break;

    end

end

%% 10. SAVE LAST-EPOCH NETWORK

lastEpoch = epoch;
lastEpochNet = fineTuneNet;

save(lastEpochPath, ...
    'lastEpochNet', ...
    'lastEpoch', ...
    'trainingLossHistory', ...
    'validationAccuracyHistory', ...
    'poultryClasses');

%% 11. RESTORE THE BEST VALIDATION CHECKPOINT

if isempty(bestNet)
    error('No valid best model was produced.');
end

fineTunedNet = bestNet;

fprintf('\n');
fprintf('====================================================\n');
fprintf(' RESTORING BEST VALIDATION MODEL\n');
fprintf('====================================================\n');

fprintf('Best epoch: %d\n', bestEpoch);
fprintf('Best validation accuracy: %.2f%%\n', ...
    100 * bestValidationAccuracy);

%% 12. FINAL PARAMETER VERIFICATION

finalLearnables = fineTunedNet.Learnables;

for k = 1:numberFrozen

    finalValue = extractdata( ...
        finalLearnables.Value{frozenIndices(k)});

    if ~isequal(finalValue, frozenParameterSnapshot{k})
        error('Final model contains a changed frozen parameter.');
    end

end

finalState = fineTunedNet.State;

for k = 1:height(finalState)

    finalValue = extractdataIfNeeded(finalState.Value{k});

    if ~isequal(finalValue, stateSnapshot{k})
        error('Final model Batch Normalization state changed.');
    end

end

fprintf('Final frozen-parameter verification: PASSED\n');
fprintf('Final Batch Normalization verification: PASSED\n');

%% 13. SAVE FINAL BEST MODEL

save(finalModelPath, ...
    'fineTunedNet', ...
    'bestEpoch', ...
    'bestValidationAccuracy', ...
    'poultryClasses', ...
    'trainingLossHistory', ...
    'validationAccuracyHistory', ...
    'globalLearnRate', ...
    'miniBatchSize', ...
    'selectedLayers');

%% 14. FINAL TRAINING REPORT

fprintf('\n');
fprintf('====================================================\n');
fprintf(' STEP 6T COMPLETED\n');
fprintf('====================================================\n');

fprintf('Best epoch: %d\n', bestEpoch);
fprintf('Best validation accuracy: %.2f%%\n', ...
    100 * bestValidationAccuracy);

fprintf('Trainable parameter groups: %d\n', numberTrainable);
fprintf('Frozen parameter groups: %d\n', numberFrozen);

fprintf('\nFILES SAVED:\n');
fprintf('%s\n', bestCheckpointPath);
fprintf('%s\n', finalModelPath);
fprintf('%s\n', lastEpochPath);
fprintf('%s\n', historyCsvPath);

fprintf('\nThe 55 test recordings were not used.\n');
fprintf('STEP 6T FINISHED SUCCESSFULLY.\n');

%% LOCAL FUNCTION 1: MODEL LOSS AND GRADIENTS

function [loss, gradients] = modelGradients(net, dlX, dlTargets)

    scores = forward(net, dlX);

    % Three-class categorical cross-entropy.
    loss = -mean(sum( ...
        dlTargets .* log(scores + 1e-7), 1));

    gradients = dlgradient(loss, net.Learnables);

end

%% LOCAL FUNCTION 2: LOAD AND PREPROCESS ONE RECORDING

function spectrograms = loadRecordingSpectrograms( ...
    audioFile, targetSampleRate, minimumDuration)

    if ~isfile(audioFile)
        error('Audio file not found: %s', audioFile);
    end

    [audio, originalSampleRate] = audioread(audioFile);

    if isempty(audio)
        error('Empty audio recording: %s', audioFile);
    end

    % Convert stereo or multichannel audio to mono.
    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end

    audio = single(audio);

    % Resample to 16 kHz when necessary.
    if originalSampleRate ~= targetSampleRate
        audio = resample( ...
            audio, targetSampleRate, originalSampleRate);
    end

    % Pad short recordings in memory only.
    minimumSamples = round(minimumDuration * targetSampleRate);

    if numel(audio) < minimumSamples
        audio(end+1:minimumSamples, 1) = 0;
    end

    % Generate YAMNet input windows.
    spectrograms = yamnetPreprocess( ...
        audio, targetSampleRate);

    spectrograms = single(spectrograms);

    if ndims(spectrograms) ~= 4 || ...
            size(spectrograms, 1) ~= 96 || ...
            size(spectrograms, 2) ~= 64 || ...
            size(spectrograms, 3) ~= 1 || ...
            size(spectrograms, 4) < 1

        error('Unexpected YAMNet input dimensions for: %s', audioFile);

    end

end

%% LOCAL FUNCTION 3: RECORDING PATH RESOLUTION

function audioFile = resolveAudioPath(fileValue, projectRoot)

    audioFile = char(string(fileValue));

    if isfile(audioFile)
        return;
    end

    audioFile = fullfile(projectRoot, char(string(fileValue)));

    if ~isfile(audioFile)
        error('Unable to resolve audio path: %s', char(string(fileValue)));
    end

end

%% LOCAL FUNCTION 4: RECORDING-LEVEL VALIDATION

function [accuracy, predictionTable] = evaluateRecordingLevel( ...
    net, validationTable, poultryClasses, projectRoot, ...
    targetSampleRate, minimumDuration)

    numberRecordings = height(validationTable);
    numberClasses = numel(poultryClasses);

    actualLabels = strings(numberRecordings, 1);
    predictedLabels = strings(numberRecordings, 1);
    averageScores = zeros(numberRecordings, numberClasses);

    for r = 1:numberRecordings

        audioFile = resolveAudioPath( ...
            validationTable.FileName(r), projectRoot);

        spectrograms = loadRecordingSpectrograms( ...
            audioFile, targetSampleRate, minimumDuration);

        numberWindows = size(spectrograms, 4);
        totalScores = zeros(numberClasses, 1);

        % Process windows in batches to control memory use.
        validationBatchSize = 64;

        for firstWindow = 1:validationBatchSize:numberWindows

            lastWindow = min( ...
                firstWindow + validationBatchSize - 1, ...
                numberWindows);

            windowIndices = firstWindow:lastWindow;

            batch = spectrograms(:, :, :, windowIndices);

            dlX = dlarray(single(batch), 'SSCB');

            scores = predict(net, dlX);

            scoreValues = double(extractdata(scores));
            scoreValues = reshape(scoreValues, numberClasses, []);

            if size(scoreValues, 2) ~= numel(windowIndices)
                error('Unexpected validation score dimensions.');
            end

            totalScores = totalScores + sum(scoreValues, 2);

        end

        % Average all window predictions into one recording prediction.
        recordingScores = totalScores / numberWindows;

        averageScores(r, :) = recordingScores(:)';

        [~, predictedIndex] = max(recordingScores);

        predictedLabels(r) = poultryClasses(predictedIndex);
        actualLabels(r) = string(validationTable.ClassLabel(r));

        if mod(r, 10) == 0 || r == numberRecordings
            fprintf('Validated %d/%d recordings.\n', ...
                r, numberRecordings);
        end

    end

    accuracy = mean(predictedLabels == actualLabels);

    predictionTable = table( ...
        string(validationTable.FileName), ...
        actualLabels, ...
        predictedLabels, ...
        'VariableNames', { ...
        'FileName', 'ActualClass', 'PredictedClass'});

    for c = 1:numberClasses

        scoreColumnName = matlab.lang.makeValidName( ...
            "Score_" + poultryClasses(c));

        predictionTable.(scoreColumnName) = averageScores(:, c);

    end

end

%% LOCAL FUNCTION 5: EXTRACT NUMERIC STATE VALUES

function value = extractdataIfNeeded(value)

    if isa(value, 'dlarray')
        value = extractdata(value);
    end

end
