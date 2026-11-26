
%% STEP 6J: TEST SELECTIVE GRADIENT CONTROL
clear; clc;

fprintf('\n============================================\n');
fprintf('STEP 6J: SELECTIVE GRADIENT CONTROL TEST\n');
fprintf('============================================\n\n');

%% 1. Load the trained frozen-head model
modelFile = fullfile(pwd, ...
    'Results','YAMNet_FrozenHead_Trained.mat');

assert(isfile(modelFile), ...
    'Frozen-head model file not found.');

loadedData = load(modelFile);
fineTuneNet = loadedData.fineTuneNet;

fprintf('Frozen-head model loaded successfully.\n');

%% 2. Load training manifest
manifestFile = fullfile(pwd, ...
    'Dataset','Dataset_Splits', ...
    'Dataset_Split_Manifest.csv');

assert(isfile(manifestFile), ...
    'Dataset manifest not found.');

fid = fopen(manifestFile,'r');
assert(fid ~= -1,'Could not open manifest.');

header = fgetl(fid);
fileNames = {};
classLabels = {};
datasetSplits = {};

while ~feof(fid)
    line = fgetl(fid);

    if ~ischar(line) || isempty(strtrim(line))
        continue;
    end

    parts = regexp(line, ...
        '^(.*),([^,]+),([^,]+)$','tokens','once');

    if isempty(parts)
        continue;
    end

    fileNames{end+1,1} = strtrim(parts{1});
    classLabels{end+1,1} = strtrim(parts{2});
    datasetSplits{end+1,1} = strtrim(parts{3});
end

fclose(fid);

T = table(fileNames,classLabels,datasetSplits, ...
    'VariableNames', ...
    {'FileName','ClassLabel','DatasetSplit'});

%% 3. Select one training recording per class
classOrder = [
    "Background_Noise"
    "Healthy"
    "Respiratory_Abnormal"
];

selectedFiles = strings(3,1);

for c = 1:3
    idx = find( ...
        string(T.ClassLabel) == classOrder(c) & ...
        string(T.DatasetSplit) == "Training",1);

    assert(~isempty(idx), ...
        'No training recording found for a class.');

    selectedFiles(c) = string(T.FileName{idx});
end

%% 4. Prepare one spectrogram window per class
X = zeros(96,64,1,3,'single');

for c = 1:3

    audioFile = char(selectedFiles(c));

    if ~isfile(audioFile)
        audioFile = fullfile(pwd,audioFile);
    end

    assert(isfile(audioFile), ...
        'Audio file not found: %s',audioFile);

    [audioIn,fs] = audioread(audioFile);

    if size(audioIn,2) > 1
        audioIn = mean(audioIn,2);
    end

    if fs ~= 16000
        audioIn = resample(audioIn,16000,fs);
        fs = 16000;
    end

    if numel(audioIn) < round(1.5*fs)
        audioIn(end+1:round(1.5*fs)) = 0;
    end

    spectrograms = yamnetPreprocess(audioIn,fs);

    assert(size(spectrograms,1) == 96 && ...
           size(spectrograms,2) == 64, ...
           'Unexpected YAMNet input dimensions.');

    X(:,:,:,c) = single(spectrograms(:,:,:,1));

    fprintf('Prepared: %s\n',classOrder(c));
end

%% 5. Calculate gradients
dlX = dlarray(X,'SSCB');

[loss,gradients] = dlfeval( ...
    @selectiveGradientFunction,fineTuneNet,dlX);

fprintf('\nGradient test loss: %.6f\n', ...
    double(gather(extractdata(loss))));

%% 6. Inspect gradient groups
learnables = fineTuneNet.Learnables;

selectedLayers = ["conv2d_13","dense"];

fprintf('\nSELECTED LAYER GRADIENTS:\n');

for i = 1:height(learnables)

    layerName = string(learnables.Layer(i));
    parameterName = string(learnables.Parameter(i));

    gradValue = gradients.Value{i};

    gradNorm = norm(double(gather( ...
        extractdata(gradValue(:)))));

    if any(layerName == selectedLayers)
        fprintf('%-15s %-10s Gradient norm: %.8f\n', ...
            layerName,parameterName,gradNorm);
    end
end

%% 7. Verify selected gradients are nonzero
for layerName = selectedLayers

    rows = find(string(learnables.Layer) == layerName);

    assert(~isempty(rows), ...
        'Selected layer not found.');

    for r = rows'

        gradNorm = norm(double(gather( ...
            extractdata(gradients.Value{r}(:)))));

        assert(isfinite(gradNorm) && gradNorm > 0, ...
            'Selected gradient is zero or invalid: %s', ...
            layerName);
    end
end

fprintf('\nBoth selected layers have valid nonzero gradients.\n');

%% 8. Confirm no model weights were updated
fprintf('No optimizer or weight update was performed.\n');

fprintf('\n============================================\n');
fprintf('STEP 6J GRADIENT TEST COMPLETED\n');
fprintf('============================================\n');

%% Local gradient function
function [loss,gradients] = ...
    selectiveGradientFunction(net,dlX)

    scores = forward(net,dlX);

    targets = dlarray(single(eye(3)),'CB');

    loss = crossentropy(scores,targets);

    gradients = dlgradient(loss,net.Learnables);
end
