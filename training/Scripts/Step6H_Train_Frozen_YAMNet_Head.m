
%% STEP 6H: TRAIN THE THREE-CLASS YAMNET CLASSIFICATION HEAD
% The pretrained YAMNet feature extractor remains frozen.
% Only dense/Weights and dense/Bias are updated.
%
% Training:
%   240 training recordings
%   50 validation recordings
%   30 maximum epochs
%   Mini-batch size: 16
%   Optimizer: Adam
%
% IMPORTANT:
% The pretrained layers receive zero gradients.
% Only the new three-class dense layer is updated.

clear;
clc;

fprintf('\n============================================\n');
fprintf('STEP 6H: FROZEN YAMNET HEAD TRAINING\n');
fprintf('============================================\n\n');

%% 1. PROJECT PATHS AND TRAINING SETTINGS

projectFolder = pwd;

modelFile = fullfile(projectFolder, ...
    'Results', 'YAMNet_ThreeClass_Initialized.mat');

manifestFile = fullfile(projectFolder, ...
    'Dataset', 'Dataset_Splits', 'Dataset_Split_Manifest.csv');

resultsFolder = fullfile(projectFolder,'Results');

assert(isfile(modelFile), ...
    'Initialized three-class model not found.');

assert(isfile(manifestFile), ...
    'Dataset manifest not found.');

if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

rng(42,'twister');

poultryClasses = [
    "Background_Noise"
    "Healthy"
    "Respiratory_Abnormal"
];

numClasses = 3;

numEpochs = 30;
batchSize = 16;

learnRate = 0.001;
patienceLimit = 7;

%% 2. LOAD THE INITIALIZED THREE-CLASS NETWORK

loadedData = load(modelFile);

assert(isfield(loadedData,'fineTuneNet'), ...
    'Variable fineTuneNet was not found.');

fineTuneNet = loadedData.fineTuneNet;

fprintf('Initialized three-class YAMNet loaded successfully.\n');

%% 3. READ THE DATASET MANIFEST

lines = readlines(manifestFile);

fileNames = strings(0,1);
classLabels = strings(0,1);
datasetSplits = strings(0,1);

for i = 2:numel(lines)

    line = strtrim(lines(i));

    if line == ""
        continue;
    end

    parts = regexp(char(line), ...
        '^(.*),([^,]+),([^,]+)$', ...
        'tokens','once');

    if isempty(parts)
        error('Manifest parsing failed at line %d.',i);
    end

    fileNames(end+1,1) = string(parts{1});
    classLabels(end+1,1) = string(parts{2});
    datasetSplits(end+1,1) = string(parts{3});

end

%% 4. SEPARATE TRAINING AND VALIDATION RECORDINGS

trainIndices = find(datasetSplits == "Training");
valIndices = find(datasetSplits == "Validation");

assert(numel(trainIndices) == 240, ...
    'Expected exactly 240 training recordings.');

assert(numel(valIndices) == 50, ...
    'Expected exactly 50 validation recordings.');

trainLabels = zeros(numel(trainIndices),1);

for i = 1:numel(trainIndices)

    classIndex = find( ...
        poultryClasses == classLabels(trainIndices(i)),1);

    assert(~isempty(classIndex), ...
        'Unknown training class.');

    trainLabels(i) = classIndex;

end

valLabels = zeros(numel(valIndices),1);

for i = 1:numel(valIndices)

    classIndex = find( ...
        poultryClasses == classLabels(valIndices(i)),1);

    assert(~isempty(classIndex), ...
        'Unknown validation class.');

    valLabels(i) = classIndex;

end

fprintf('\nTraining recordings: %d\n',numel(trainIndices));
fprintf('Validation recordings: %d\n',numel(valIndices));

%% 5. IDENTIFY THE NEW DENSE CLASSIFICATION LAYER

learnables = fineTuneNet.Learnables;

denseRows = find( ...
    string(learnables.Layer) == "dense");

weightRow = find( ...
    string(learnables.Layer) == "dense" & ...
    string(learnables.Parameter) == "Weights");

biasRow = find( ...
    string(learnables.Layer) == "dense" & ...
    string(learnables.Parameter) == "Bias");

assert(numel(denseRows) == 2, ...
    'Expected exactly two dense-layer learnable parameters.');

assert(numel(weightRow) == 1, ...
    'Dense weights were not identified correctly.');

assert(numel(biasRow) == 1, ...
    'Dense bias was not identified correctly.');

fprintf('\nNew classification head identified.\n');

fprintf('Dense weights: %s\n', ...
    mat2str(size(learnables.Value{weightRow})));

fprintf('Dense bias: %s\n', ...
    mat2str(size(learnables.Value{biasRow})));

%% 6. SAVE THE ORIGINAL PRETRAINED PARAMETERS FOR VERIFICATION

% Store all learnable values except the new dense layer.
% These values must remain unchanged throughout training.

originalLearnables = fineTuneNet.Learnables;

pretrainedRows = setdiff( ...
    (1:height(originalLearnables))',denseRows);

originalPretrainedValues = cell(numel(pretrainedRows),1);

for i = 1:numel(pretrainedRows)

    row = pretrainedRows(i);

    originalPretrainedValues{i} = ...
        extractdata(originalLearnables.Value{row});

end

fprintf('\nPretrained parameter groups: %d\n', ...
    numel(pretrainedRows));

fprintf('New classification parameter groups: %d\n', ...
    numel(denseRows));

%% 7. INITIALIZE ADAM OPTIMIZER

averageGradient = [];
averageSquaredGradient = [];

iteration = 0;

%% 8. INITIALIZE TRAINING HISTORY

trainingLossHistory = nan(numEpochs,1);

validationAccuracyHistory = nan(numEpochs,1);

bestValidationAccuracy = -inf;

bestEpoch = 0;

epochsWithoutImprovement = 0;

bestNet = fineTuneNet;

%% 9. BEGIN TRAINING

for epoch = 1:numEpochs

    fprintf('\n--------------------------------------------\n');
    fprintf('EPOCH %d OF %d\n',epoch,numEpochs);
    fprintf('--------------------------------------------\n');

    %% 9A. PREPARE ONE RANDOM WINDOW PER TRAINING RECORDING

    Xtrain = zeros(96,64,1,numel(trainIndices),'single');

    for r = 1:numel(trainIndices)

        audioPath = char(fileNames(trainIndices(r)));

        assert(isfile(audioPath), ...
            'Training audio not found: %s',audioPath);

        [audioIn,fs] = audioread(audioPath);

        % Convert stereo to mono.
        if size(audioIn,2) > 1
            audioIn = mean(audioIn,2);
        end

        % Resample to 16 kHz.
        if fs ~= 16000
            audioIn = resample(audioIn,16000,fs);
            fs = 16000;
        end

        % Pad short recordings in memory only.
        minimumSamples = round(1.5*fs);

        if numel(audioIn) < minimumSamples
            audioIn(end+1:minimumSamples,1) = 0;
        end

        % Generate YAMNet spectrogram windows.
        spectrograms = yamnetPreprocess(audioIn,fs);

        numberOfWindows = size(spectrograms,4);

        assert(numberOfWindows >= 1, ...
            'No YAMNet windows generated.');

        % Randomly select one window for this recording.
        selectedWindow = randi(numberOfWindows);

        Xtrain(:,:,:,r) = single( ...
            spectrograms(:,:,1,selectedWindow));

    end

    %% 9B. SHUFFLE TRAINING RECORDINGS

    shuffledIndices = randperm(numel(trainIndices));

    epochLoss = 0;
    numberOfBatches = 0;

    %% 9C. MINI-BATCH TRAINING

    for startIndex = 1:batchSize:numel(shuffledIndices)

        endIndex = min( ...
            startIndex+batchSize-1, ...
            numel(shuffledIndices));

        batchIndices = shuffledIndices(startIndex:endIndex);

        Xbatch = Xtrain(:,:,:,batchIndices);

        currentBatchSize = numel(batchIndices);

        Ybatch = zeros( ...
            numClasses,currentBatchSize,'single');

        for j = 1:currentBatchSize

            originalIndex = batchIndices(j);

            classIndex = trainLabels(originalIndex);

            Ybatch(classIndex,j) = 1;

        end

        dlX = dlarray(Xbatch,'SSCB');

        %% 9D. FORWARD PASS AND GRADIENT CALCULATION

        [loss,gradients] = dlfeval( ...
            @modelGradients, ...
            fineTuneNet, ...
            dlX, ...
            Ybatch);

        %% 9E. FREEZE ALL PRETRAINED LAYERS

        % Set every pretrained parameter gradient to zero.
        % Only the new dense-layer gradients remain active.

        for i = 1:numel(pretrainedRows)

            row = pretrainedRows(i);

            gradientValue = gradients.Value{row};

            gradients.Value{row} = zeros( ...
                size(gradientValue),'like',gradientValue);

        end

        %% 9F. UPDATE NETWORK USING ADAM

        iteration = iteration + 1;

        [fineTuneNet, ...
            averageGradient, ...
            averageSquaredGradient] = adamupdate( ...
            fineTuneNet, ...
            gradients, ...
            averageGradient, ...
            averageSquaredGradient, ...
            iteration, ...
            learnRate);

        epochLoss = epochLoss + ...
            double(extractdata(loss));

        numberOfBatches = numberOfBatches + 1;

    end

    %% 9G. CALCULATE AVERAGE TRAINING LOSS

    averageEpochLoss = epochLoss/numberOfBatches;

    trainingLossHistory(epoch) = averageEpochLoss;

    %% 9H. VERIFY PRETRAINED PARAMETERS REMAIN UNCHANGED

    currentLearnables = fineTuneNet.Learnables;

    for i = 1:numel(pretrainedRows)

        row = pretrainedRows(i);

        currentValue = extractdata( ...
            currentLearnables.Value{row});

        originalValue = originalPretrainedValues{i};

        assert(isequal(currentValue,originalValue), ...
            'A pretrained parameter changed during training.');

    end

    %% 9I. EVALUATE VALIDATION RECORDINGS

    [validationAccuracy,~,~] = evaluateRecordings( ...
        fineTuneNet, ...
        fileNames(valIndices), ...
        valLabels, ...
        projectFolder, ...
        poultryClasses);

    validationAccuracyHistory(epoch) = validationAccuracy;

    fprintf('\nTraining loss: %.6f\n',averageEpochLoss);

    fprintf('Validation accuracy: %.2f%%\n', ...
        validationAccuracy*100);

    fprintf('Pretrained weights verification: PASSED\n');

    %% 9J. SAVE BEST VALIDATION CHECKPOINT

    if validationAccuracy > bestValidationAccuracy

        bestValidationAccuracy = validationAccuracy;

        bestEpoch = epoch;

        bestNet = fineTuneNet;

        epochsWithoutImprovement = 0;

        save(fullfile(resultsFolder, ...
            'YAMNet_FrozenHead_Best_Checkpoint.mat'), ...
            'bestNet', ...
            'bestValidationAccuracy', ...
            'bestEpoch', ...
            'poultryClasses', ...
            'trainingLossHistory', ...
            'validationAccuracyHistory', ...
            '-v7.3');

        fprintf('Best validation checkpoint saved.\n');

    else

        epochsWithoutImprovement = ...
            epochsWithoutImprovement + 1;

    end

    %% 9K. EARLY STOPPING

    if epochsWithoutImprovement >= patienceLimit

        fprintf('\nEarly stopping triggered.\n');

        break;

    end

end

%% 10. RESTORE THE BEST VALIDATION MODEL

fineTuneNet = bestNet;

fprintf('\n============================================\n');
fprintf('FROZEN HEAD TRAINING FINISHED\n');
fprintf('============================================\n');

fprintf('Best epoch: %d\n',bestEpoch);

fprintf('Best validation accuracy: %.2f%%\n', ...
    bestValidationAccuracy*100);

%% 11. VERIFY THE SAVED BEST MODEL

bestLearnables = fineTuneNet.Learnables;

for i = 1:numel(pretrainedRows)

    row = pretrainedRows(i);

    bestValue = extractdata(bestLearnables.Value{row});

    originalValue = originalPretrainedValues{i};

    assert(isequal(bestValue,originalValue), ...
        'A pretrained parameter changed in the best model.');

end

fprintf('\nFinal pretrained-weight verification: PASSED\n');

%% 12. SAVE TRAINING HISTORY

completedEpochs = find(~isnan(trainingLossHistory));

trainingHistory = table( ...
    completedEpochs, ...
    trainingLossHistory(completedEpochs), ...
    validationAccuracyHistory(completedEpochs), ...
    'VariableNames', ...
    {'Epoch','TrainingLoss','ValidationAccuracy'});

writetable(trainingHistory, ...
    fullfile(resultsFolder, ...
    'YAMNet_FrozenHead_Training_History.csv'));

%% 13. SAVE TRAINED MODEL

save(fullfile(resultsFolder, ...
    'YAMNet_FrozenHead_Trained.mat'), ...
    'fineTuneNet', ...
    'bestValidationAccuracy', ...
    'bestEpoch', ...
    'poultryClasses', ...
    'trainingHistory', ...
    '-v7.3');

%% 14. COMPLETION MESSAGE

fprintf('\nSaved trained model:\n');
fprintf('Results\\YAMNet_FrozenHead_Trained.mat\n');

fprintf('\nSaved training history:\n');
fprintf('Results\\YAMNet_FrozenHead_Training_History.csv\n');

fprintf('\n============================================\n');
fprintf('STEP 6H COMPLETED SUCCESSFULLY\n');
fprintf('============================================\n');

%% LOCAL FUNCTION 1: GRADIENT CALCULATION

function [loss,gradients] = modelGradients( ...
    net,dlX,oneHotLabels)

    % Forward pass through YAMNet.
    scores = forward(net,dlX);

    batchSize = size(oneHotLabels,2);

    % Arrange predictions as classes x batch.
    scores = reshape(stripdims(scores), ...
        [3,batchSize]);

    targets = dlarray(single(oneHotLabels));

    % Three-class cross-entropy loss.
    loss = -mean(sum( ...
        targets .* log(scores+1e-7),1),'all');

    % Calculate gradients for all network parameters.
    gradients = dlgradient(loss,net.Learnables);

end

%% LOCAL FUNCTION 2: RECORDING-LEVEL VALIDATION

function [accuracy,predictedLabels,recordingScores] = ...
    evaluateRecordings( ...
    net,validationFiles,trueLabels, ...
    projectFolder,poultryClasses)

    numRecordings = numel(validationFiles);

    numClasses = numel(poultryClasses);

    recordingScores = zeros(numRecordings,numClasses);

    for r = 1:numRecordings

        audioPath = char(validationFiles(r));

        if ~isfile(audioPath)
            audioPath = fullfile(projectFolder,audioPath);
        end

        assert(isfile(audioPath), ...
            'Validation audio not found: %s',audioPath);

        [audioIn,fs] = audioread(audioPath);

        % Convert stereo to mono.
        if size(audioIn,2) > 1
            audioIn = mean(audioIn,2);
        end

        % Resample to 16 kHz.
        if fs ~= 16000
            audioIn = resample(audioIn,16000,fs);
            fs = 16000;
        end

        % Pad short recordings in memory.
        minimumSamples = round(1.5*fs);

        if numel(audioIn) < minimumSamples
            audioIn(end+1:minimumSamples,1) = 0;
        end

        % Generate spectrogram windows.
        spectrograms = yamnetPreprocess(audioIn,fs);

        numberOfWindows = size(spectrograms,4);

        windowScores = zeros(numberOfWindows,numClasses);

        validationBatchSize = 64;

        %% Process windows in batches.

        for firstWindow = 1:validationBatchSize:numberOfWindows

            lastWindow = min( ...
                firstWindow+validationBatchSize-1, ...
                numberOfWindows);

            currentBatch = single( ...
                spectrograms(:,:,:,firstWindow:lastWindow));

            batchScores = predict(net,currentBatch);

            batchScores = squeeze(batchScores);

            currentBatchSize = lastWindow-firstWindow+1;

            % Ensure rows represent windows and
            % columns represent the three classes.

            if size(batchScores,1) ~= currentBatchSize

                batchScores = batchScores';

            end

            assert(size(batchScores,1) == currentBatchSize && ...
                   size(batchScores,2) == numClasses, ...
                'Unexpected validation prediction dimensions.');

            windowScores(firstWindow:lastWindow,:) = ...
                double(batchScores);

        end

        % Average all window predictions into one
        % recording-level prediction.

        recordingScores(r,:) = mean(windowScores,1);

    end

    % Final recording-level classification.
    [~,predictedLabels] = max(recordingScores,[],2);

    accuracy = mean(predictedLabels == trueLabels);

end
