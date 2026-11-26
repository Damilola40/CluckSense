
%% STEP 6S: TEMPORARY MINIBATCH UPDATE TEST
% Tests one real minibatch update without modifying saved models.

clear; clc;

fprintf('\n============================================\n');
fprintf('STEP 6S: TEMPORARY MINIBATCH UPDATE TEST\n');
fprintf('============================================\n\n');

rng(42);

projectRoot = pwd;
targetSampleRate = 16000;
minimumDuration = 1.5;
batchSize = 4;

classOrder = [
    "Background_Noise"
    "Healthy"
    "Respiratory_Abnormal"
];

%% 1. LOAD BEST CHECKPOINT AND DATA MANIFEST

checkpointFile = fullfile(projectRoot,'Results', ...
    'YAMNet_FrozenHead_Best_Checkpoint.mat');

inspectionFile = fullfile(projectRoot,'Results', ...
    'FineTuning_Data_Inspection.mat');

assert(isfile(checkpointFile),'Best checkpoint not found.');
assert(isfile(inspectionFile),'Fine-tuning inspection file not found.');

C = load(checkpointFile);
D = load(inspectionFile);

testNet = C.bestNet;

assert(isa(testNet,'dlnetwork'));
assert(testNet.Initialized);

assert(isfield(D,'T'),'Training manifest table T not found.');

T = D.T;

trainingRows = find(string(T.DatasetSplit) == "Training");

assert(numel(trainingRows) == 240, ...
    'Expected exactly 240 training recordings.');

fprintf('Best checkpoint loaded.\n');
fprintf('Training recordings available: %d\n', ...
    numel(trainingRows));

%% 2. CONFIGURE SELECTIVE LEARNING-RATE FACTORS

L = testNet.Learnables;

% Freeze all 110 learnable parameter groups.
for i = 1:height(L)

    layerName = string(L.Layer(i));
    parameterName = string(L.Parameter(i));

    testNet = setLearnRateFactor( ...
        testNet,layerName,parameterName,0);
end

% Unfreeze final convolutional layer.
testNet = setLearnRateFactor( ...
    testNet,"conv2d_13","Weights",0.1);

testNet = setLearnRateFactor( ...
    testNet,"conv2d_13","Bias",0.1);

% Unfreeze three-class classification layer.
testNet = setLearnRateFactor( ...
    testNet,"dense","Weights",1.0);

testNet = setLearnRateFactor( ...
    testNet,"dense","Bias",1.0);

%% 3. SELECT FOUR REAL TRAINING RECORDINGS

selectedRows = zeros(batchSize,1);

% Select one recording from each class.
for k = 1:3

    classRows = trainingRows( ...
        string(T.ClassLabel(trainingRows)) == classOrder(k));

    assert(~isempty(classRows), ...
        'A required training class has no recordings.');

    selectedRows(k) = classRows(randi(numel(classRows)));
end

% Select one additional training recording.
selectedRows(4) = trainingRows( ...
    randi(numel(trainingRows)));

fprintf('\nSELECTED TRAINING RECORDINGS:\n');

%% 4. PREPROCESS THE FOUR RECORDINGS

X = zeros(96,64,1,batchSize,'single');
targetMatrix = zeros(3,batchSize,'single');

for k = 1:batchSize

    row = selectedRows(k);

    fileName = string(T.FileName(row));
    classLabel = string(T.ClassLabel(row));

    audioFile = char(fileName);

    % Resolve relative paths if necessary.
    if ~isfile(audioFile)

        candidate = fullfile(projectRoot, ...
            'Dataset','Processed_Audio', ...
            char(classLabel),char(fileName));

        if isfile(candidate)
            audioFile = candidate;
        end
    end

    assert(isfile(audioFile), ...
        'Audio file not found: %s',audioFile);

    [audioIn,fs] = audioread(audioFile);

    % Convert stereo audio to mono.
    if size(audioIn,2) > 1
        audioIn = mean(audioIn,2);
    end

    % Resample to 16 kHz.
    if fs ~= targetSampleRate
        audioIn = resample(audioIn, ...
            targetSampleRate,fs);
    end

    audioIn = single(audioIn(:));

    % Pad short recordings in memory only.
    minimumSamples = round( ...
        minimumDuration*targetSampleRate);

    if numel(audioIn) < minimumSamples
        audioIn(end+1:minimumSamples,1) = 0;
    end

    % Generate YAMNet spectrogram windows.
    spectrograms = yamnetPreprocess( ...
        audioIn,targetSampleRate);

    assert(size(spectrograms,1) == 96);
    assert(size(spectrograms,2) == 64);
    assert(size(spectrograms,3) == 1);

    % Select one random window from this recording.
    windowIndex = randi(size(spectrograms,4));

    X(:,:,:,k) = single( ...
        spectrograms(:,:,:,windowIndex));

    classIndex = find(classOrder == classLabel,1);

    assert(~isempty(classIndex), ...
        'Unknown class label: %s',classLabel);

    targetMatrix(classIndex,k) = 1;

    fprintf('%d. %-22s | %s | Window %d of %d\n', ...
        k,classLabel,fileName, ...
        windowIndex,size(spectrograms,4));
end

%% 5. PREPARE MINIBATCH

dlX = dlarray(X,'SSCB');
dlTargets = dlarray(targetMatrix);

fprintf('\nMINIBATCH DIMENSIONS:\n');
fprintf('Input: %s\n',mat2str(size(X)));
fprintf('Targets: %s\n',mat2str(size(targetMatrix)));

%% 6. SAVE BEFORE-UPDATE SNAPSHOTS

beforeLearnables = testNet.Learnables;
beforeState = testNet.State;

%% 7. CALCULATE LOSS AND GRADIENTS

[loss,gradients] = dlfeval( ...
    @modelGradients,testNet,dlX,dlTargets);

lossValue = double(extractdata(loss));

fprintf('\nMINIBATCH LOSS: %.8f\n',lossValue);

assert(isfinite(lossValue), ...
    'Loss is not finite.');

%% 8. EXPLICITLY FREEZE NON-SELECTED GRADIENTS

for i = 1:height(gradients)

    layerName = string(gradients.Layer(i));
    parameterName = string(gradients.Parameter(i));

    factor = getLearnRateFactor( ...
        testNet,layerName,parameterName);

    if factor == 0

        gradients.Value{i} = zeros( ...
            size(gradients.Value{i}), ...
            'like',gradients.Value{i});
    end
end

%% 9. PERFORM ONE TEMPORARY ADAM UPDATE

globalLearningRate = 0.0001;

[testNet,averageGrad,averageSqGrad] = adamupdate( ...
    testNet,gradients,[],[],1,globalLearningRate);

fprintf('\nTemporary Adam update completed.\n');

%% 10. VERIFY SELECTED PARAMETER CHANGES

afterLearnables = testNet.Learnables;

selectedLayers = ["conv2d_13","dense"];

fprintf('\nSELECTED LAYER UPDATE VERIFICATION:\n');

for k = 1:numel(selectedLayers)

    layerName = selectedLayers(k);

    rows = string(beforeLearnables.Layer) == layerName;

    layerChanged = false;

    for j = find(rows)'

        beforeValue = extractdata( ...
            beforeLearnables.Value{j});

        afterValue = extractdata( ...
            afterLearnables.Value{j});

        difference = norm(double( ...
            afterValue(:)-beforeValue(:)));

        fprintf('%s %s difference: %.12g\n', ...
            layerName, ...
            string(beforeLearnables.Parameter(j)), ...
            difference);

        if difference > 0
            layerChanged = true;
        end
    end

    assert(layerChanged, ...
        'No parameter changed in selected layer %s.', ...
        layerName);
end

%% 11. VERIFY ALL FROZEN PARAMETERS

frozenGroupsUnchanged = true;

for i = 1:height(beforeLearnables)

    layerName = string(beforeLearnables.Layer(i));
    parameterName = string(beforeLearnables.Parameter(i));

    factor = getLearnRateFactor( ...
        testNet,layerName,parameterName);

    if factor == 0

        beforeValue = extractdata( ...
            beforeLearnables.Value{i});

        afterValue = extractdata( ...
            afterLearnables.Value{i});

        if ~isequal(beforeValue,afterValue)
            frozenGroupsUnchanged = false;

            fprintf('FROZEN PARAMETER CHANGED: %s / %s\n', ...
                layerName,parameterName);
        end
    end
end

assert(frozenGroupsUnchanged, ...
    'One or more frozen parameters changed.');

fprintf('All frozen learnable parameters remain unchanged.\n');

%% 12. VERIFY BATCH NORMALIZATION STATE

afterState = testNet.State;

stateUnchanged = true;

assert(height(beforeState) == height(afterState));

for i = 1:height(beforeState)

    beforeValue = beforeState.Value{i};
    afterValue = afterState.Value{i};

    if ~isequal(beforeValue,afterValue)
        stateUnchanged = false;

        fprintf('STATE CHANGED: %s / %s\n', ...
            string(beforeState.Layer(i)), ...
            string(beforeState.Parameter(i)));
    end
end

assert(stateUnchanged, ...
    'Batch Normalization state changed.');

fprintf('All 54 Batch Normalization state entries remain unchanged.\n');

%% 13. VERIFY LEARNING-RATE FACTORS

fprintf('\nLEARNING-RATE FACTORS AFTER UPDATE:\n');

disp(getLearnRateFactor(testNet,"conv2d_13","Weights"));
disp(getLearnRateFactor(testNet,"conv2d_13","Bias"));
disp(getLearnRateFactor(testNet,"dense","Weights"));
disp(getLearnRateFactor(testNet,"dense","Bias"));

%% 14. FINAL REPORT

fprintf('\n============================================\n');
fprintf('STEP 6S VERIFICATION RESULTS\n');
fprintf('============================================\n');

fprintf('Temporary minibatch update: PASSED\n');
fprintf('Selected layers: PASSED\n');
fprintf('Frozen parameters: PASSED\n');
fprintf('Batch Normalization state: PASSED\n');

fprintf('\nNo model was saved or overwritten.\n');

fprintf('\n============================================\n');
fprintf('STEP 6S COMPLETED\n');
fprintf('============================================\n');

%% LOCAL GRADIENT FUNCTION

function [loss,gradients] = modelGradients(net,dlX,dlTargets)

    scores = forward(net,dlX);

    loss = crossentropy(scores,dlTargets);

    gradients = dlgradient(loss,net.Learnables);

end
