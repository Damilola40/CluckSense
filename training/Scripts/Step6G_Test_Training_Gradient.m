
%% STEP 6G: VERIFY THREE-CLASS YAMNET TRAINING GRADIENTS
% This script performs a gradient test only.
% It does NOT update or train any network weights.

clearvars -except ans
clc;

fprintf('\n============================================\n');
fprintf('STEP 6G: YAMNET GRADIENT VERIFICATION\n');
fprintf('============================================\n\n');

%% 1. Define project paths

projectFolder = pwd;

modelFile = fullfile(projectFolder, ...
    'Results', 'YAMNet_ThreeClass_Initialized.mat');

manifestFile = fullfile(projectFolder, ...
    'Dataset', 'Dataset_Splits', 'Dataset_Split_Manifest.csv');

assert(isfile(modelFile), 'Initialized YAMNet model file was not found.');
assert(isfile(manifestFile), 'Dataset manifest was not found.');

%% 2. Load initialized three-class YAMNet

loadedData = load(modelFile);

assert(isfield(loadedData, 'fineTuneNet'), ...
    'The variable fineTuneNet was not found in the saved model.');

fineTuneNet = loadedData.fineTuneNet;

fprintf('Initialized three-class YAMNet loaded successfully.\n');

%% 3. Define the three poultry classes

poultryClasses = [
    "Background_Noise"
    "Healthy"
    "Respiratory_Abnormal"
];

numberOfClasses = numel(poultryClasses);

%% 4. Read the dataset manifest safely

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
        '^(.*),([^,]+),([^,]+)$', 'tokens', 'once');

    if isempty(parts)
        error('Could not parse manifest line %d.', i);
    end

    fileNames(end+1,1) = string(parts{1});
    classLabels(end+1,1) = string(parts{2});
    datasetSplits(end+1,1) = string(parts{3});

end

%% 5. Select one training recording from each class

selectedIndices = zeros(numberOfClasses,1);

for c = 1:numberOfClasses

    matchingRows = find( ...
        datasetSplits == "Training" & ...
        classLabels == poultryClasses(c));

    assert(~isempty(matchingRows), ...
        'No training recording found for class %s.', poultryClasses(c));

    selectedIndices(c) = matchingRows(1);

end

fprintf('\nSelected training recordings:\n');

for c = 1:numberOfClasses
    fprintf('%s: %s\n', ...
        poultryClasses(c), fileNames(selectedIndices(c)));
end

%% 6. Prepare one YAMNet spectrogram window per recording

batchSize = numberOfClasses;

X = zeros(96,64,1,batchSize,'single');
oneHotLabels = zeros(numberOfClasses,batchSize,'single');

for i = 1:batchSize

    audioPath = char(fileNames(selectedIndices(i)));

    if ~isfile(audioPath)
        audioPath = fullfile(projectFolder, audioPath);
    end

    assert(isfile(audioPath), ...
        'Audio file not found: %s', audioPath);

    [audioIn, fs] = audioread(audioPath);

    % Convert stereo audio to mono.
    if size(audioIn,2) > 1
        audioIn = mean(audioIn,2);
    end

    % Resample to 16 kHz.
    if fs ~= 16000
        audioIn = resample(audioIn,16000,fs);
        fs = 16000;
    end

    % Pad short recordings in memory only.
    minimumSamples = round(1.5 * fs);

    if numel(audioIn) < minimumSamples
        audioIn(end+1:minimumSamples,1) = 0;
    end

    % Generate YAMNet input spectrograms.
    spectrograms = yamnetPreprocess(audioIn,fs);

    assert(size(spectrograms,1) == 96 && ...
           size(spectrograms,2) == 64, ...
           'Unexpected YAMNet spectrogram dimensions.');

    % Select one window for this gradient test.
    windowIndex = 1;

    X(:,:,:,i) = single( ...
        spectrograms(:,:,1,windowIndex));

    oneHotLabels(i,i) = 1;

end

fprintf('\nInput batch prepared: %d recordings.\n',batchSize);

%% 7. Convert input to a formatted dlarray

dlX = dlarray(X,'SSCB');

%% 8. Calculate loss and gradients without updating weights

[lossValue,gradients] = dlfeval( ...
    @modelGradients, fineTuneNet, dlX, oneHotLabels);

fprintf('\nGradient calculation completed.\n');
fprintf('Cross-entropy loss: %.6f\n', ...
    double(extractdata(lossValue)));

%% 9. Inspect the new classification-head gradients

learnables = fineTuneNet.Learnables;

denseRows = find(string(learnables.Layer) == "dense");

assert(numel(denseRows) == 2, ...
    'Expected exactly two learnable parameters in the dense layer.');

gradientNorms = zeros(numel(denseRows),1);

for i = 1:numel(denseRows)

    row = denseRows(i);

    gradientValue = gradients.Value{row};

    assert(~isempty(gradientValue), ...
        'A dense-layer gradient is empty.');

    gradientNorms(i) = double(extractdata( ...
        sqrt(sum(gradientValue.^2,'all'))));

end

fprintf('\nNew classification-head gradients:\n');

for i = 1:numel(denseRows)

    fprintf('%s: gradient norm = %.8f\n', ...
        string(learnables.Parameter(denseRows(i))), ...
        gradientNorms(i));

end

assert(all(isfinite(gradientNorms)), ...
    'A gradient contains an invalid value.');

assert(any(gradientNorms > 0), ...
    'The classification-head gradients are all zero.');

%% 10. Verify that no network weights were changed

% This script only calculates gradients.
% No optimizer or weight-update operation is called.

fprintf('\nNetwork weights have not been updated.\n');

fprintf('\n============================================\n');
fprintf('STEP 6G GRADIENT TEST COMPLETED SUCCESSFULLY\n');
fprintf('============================================\n');

%% LOCAL FUNCTION

function [loss,gradients] = modelGradients(net,dlX,oneHotLabels)

    % Forward pass through the complete network.
    scores = forward(net,dlX);

    % Expected output: 1 x 1 x 3 x batchSize.
    batchSize = size(oneHotLabels,2);

    scores = reshape(stripdims(scores), ...
        [3,batchSize]);

    targets = dlarray(single(oneHotLabels));

    % Three-class cross-entropy loss.
    loss = -mean(sum( ...
        targets .* log(scores + 1e-7),1),'all');

    % Calculate gradients through the network.
    gradients = dlgradient(loss,net.Learnables);

end
