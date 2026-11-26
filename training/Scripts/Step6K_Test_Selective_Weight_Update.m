
%% STEP 6K: TEST SELECTIVE WEIGHT UPDATING
clear; clc;

fprintf('\n============================================\n');
fprintf('STEP 6K: SELECTIVE WEIGHT UPDATE TEST\n');
fprintf('============================================\n\n');

%% 1. Load frozen-head model
modelFile = fullfile(pwd, ...
    'Results','YAMNet_FrozenHead_Trained.mat');

assert(isfile(modelFile),'Model file not found.');

loadedData = load(modelFile);
testNet = loadedData.fineTuneNet;

fprintf('Model loaded successfully.\n');

%% 2. Load one training recording per class
manifestFile = fullfile(pwd, ...
    'Dataset','Dataset_Splits', ...
    'Dataset_Split_Manifest.csv');

fid = fopen(manifestFile,'r');
assert(fid ~= -1,'Cannot open manifest.');

fgetl(fid);

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

classOrder = [
    "Background_Noise"
    "Healthy"
    "Respiratory_Abnormal"
];

X = zeros(96,64,1,3,'single');

for c = 1:3

    idx = find( ...
        string(T.ClassLabel) == classOrder(c) & ...
        string(T.DatasetSplit) == "Training",1);

    assert(~isempty(idx),'Training class missing.');

    audioFile = char(T.FileName{idx});

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

    spec = yamnetPreprocess(audioIn,fs);
    X(:,:,:,c) = single(spec(:,:,:,1));

    fprintf('Prepared %s recording.\n',classOrder(c));
end

dlX = dlarray(X,'SSCB');
targets = dlarray(single(eye(3)),'CB');

%% 3. Save original learnable parameters
originalLearnables = testNet.Learnables;

%% 4. Calculate gradients
[loss,gradients] = dlfeval( ...
    @selectiveUpdateGradientFunction,testNet,dlX,targets);

fprintf('\nTest loss: %.6f\n', ...
    double(gather(extractdata(loss))));

%% 5. Keep gradients only for selected layers
selectedLayers = ["conv2d_13","dense"];

for i = 1:height(gradients)

    layerName = string(gradients.Layer(i));

    if ~any(layerName == selectedLayers)
        gradients.Value{i} = ...
            zeros(size(gradients.Value{i}), ...
            'like',gradients.Value{i});
    end
end

%% 6. Perform one temporary Adam update
averageGrad = [];
averageSqGrad = [];
iteration = 1;
testLearningRate = 1e-5;

[testNet,averageGrad,averageSqGrad] = adamupdate( ...
    testNet,gradients,averageGrad,averageSqGrad, ...
    iteration,testLearningRate);

fprintf('One temporary Adam update completed.\n');

%% 7. Compare parameters before and after update
updatedLearnables = testNet.Learnables;

convChanged = false;
denseChanged = false;
frozenUnchanged = true;

fprintf('\nPARAMETER UPDATE VERIFICATION:\n\n');

for i = 1:height(originalLearnables)

    layerName = string(originalLearnables.Layer(i));
    parameterName = string(originalLearnables.Parameter(i));

    before = extractdata(originalLearnables.Value{i});
    after = extractdata(updatedLearnables.Value{i});

    difference = max(abs(double(gather(before(:))) - ...
                         double(gather(after(:)))));

    if layerName == "conv2d_13"

        fprintf('%-15s %-10s Difference: %.12g\n', ...
            layerName,parameterName,difference);

        if difference > 0
            convChanged = true;
        end

    elseif layerName == "dense"

        fprintf('%-15s %-10s Difference: %.12g\n', ...
            layerName,parameterName,difference);

        if difference > 0
            denseChanged = true;
        end

    else

        if difference ~= 0
            frozenUnchanged = false;

            fprintf('WARNING: Frozen parameter changed: %s / %s\n', ...
                layerName,parameterName);
        end
    end
end

%% 8. Final verification
assert(convChanged, ...
    'The convolution layer was not updated.');

assert(denseChanged, ...
    'The classification head was not updated.');

assert(frozenUnchanged, ...
    'One or more frozen parameters changed.');

fprintf('\nCONV2D_13: Updated successfully.\n');
fprintf('DENSE: Updated successfully.\n');
fprintf('All non-selected learnable parameters remained unchanged.\n');

fprintf('\nNo model was saved or overwritten.\n');

fprintf('\n============================================\n');
fprintf('STEP 6K SELECTIVE UPDATE TEST COMPLETED\n');
fprintf('============================================\n');

%% Local gradient function
function [loss,gradients] = ...
    selectiveUpdateGradientFunction(net,dlX,targets)

    scores = forward(net,dlX);
    loss = crossentropy(scores,targets);
    gradients = dlgradient(loss,net.Learnables);
end
