
%% STEP 6D: INITIALIZE THREE-CLASS YAMNET
% Purpose:
% 1. Load the architecture created in Step 6C.
% 2. Initialize the new three-class layer.
% 3. Verify its weights and bias.
% 4. Confirm pretrained parameters remain unchanged.
%
% No model training is performed here.

clear;
clc;

fprintf('\n============================================\n');
fprintf(' STEP 6D: INITIALIZE THREE-CLASS YAMNET\n');
fprintf('============================================\n\n');

%% 1. DEFINE PATHS

projectFolder = ...
    'C:\Users\abdul\Desktop\CluckSense_Smart_Poultry';

resultsFolder = fullfile(projectFolder, 'Results');

architecturePath = fullfile(resultsFolder, ...
    'YAMNet_ThreeClass_Architecture.mat');

if ~exist(architecturePath, 'file')
    error('Step 6C architecture file not found.');
end

%% 2. LOAD THREE-CLASS ARCHITECTURE

fprintf('Loading three-class YAMNet architecture...\n');

loadedData = load(architecturePath);

fineTuneNet = loadedData.fineTuneNet;
poultryClasses = loadedData.poultryClasses;

fprintf('Architecture loaded successfully.\n');

fprintf('Initialized before this step: %d\n', ...
    fineTuneNet.Initialized);

%% 3. INITIALIZE NETWORK

fprintf('\nInitializing network parameters...\n');

fineTuneNet = initialize(fineTuneNet);

fprintf('Network initialization completed.\n');

%% 4. VERIFY NETWORK INITIALIZATION

fprintf('\nChecking initialization status...\n');

fprintf('Initialized: %d\n', fineTuneNet.Initialized);

if ~fineTuneNet.Initialized
    error('Network initialization failed.');
end

%% 5. VERIFY NEW CLASSIFICATION LAYER

fprintf('\nInspecting new dense layer...\n');

denseLayer = getLayer(fineTuneNet, "dense");

disp(denseLayer);

if denseLayer.OutputSize ~= 3
    error('Classification layer must have three outputs.');
end

%% 6. VERIFY LEARNABLE PARAMETER DIMENSIONS

learnables = fineTuneNet.Learnables;

denseRows = learnables.Layer == "dense";

denseLearnables = learnables(denseRows,:);

disp(denseLearnables);

weightRow = find( ...
    denseRows & learnables.Parameter == "Weights");

biasRow = find( ...
    denseRows & learnables.Parameter == "Bias");

if numel(weightRow) ~= 1 || numel(biasRow) ~= 1
    error('Could not identify dense layer weights and bias.');
end

newWeights = extractdata( ...
    learnables.Value{weightRow});

newBias = extractdata( ...
    learnables.Value{biasRow});

fprintf('\nNew classification weights: %s\n', ...
    mat2str(size(newWeights)));

fprintf('New classification bias: %s\n', ...
    mat2str(size(newBias)));

if ~isequal(size(newWeights), [3 1024])
    error('Unexpected classification weight dimensions.');
end

if ~isequal(size(newBias), [3 1])
    error('Unexpected classification bias dimensions.');
end

%% 7. VERIFY PRETRAINED PARAMETERS WERE PRESERVED

fprintf('\nVerifying pretrained parameters...\n');

[originalNet, ~] = audioPretrainedNetwork("yamnet");

originalLearnables = originalNet.Learnables;

preservedCount = 0;
mismatchCount = 0;

for i = 1:height(originalLearnables)

    layerName = originalLearnables.Layer(i);
    parameterName = originalLearnables.Parameter(i);

    if layerName == "dense"
        continue;
    end

    matchingRow = find( ...
        learnables.Layer == layerName & ...
        learnables.Parameter == parameterName);

    if numel(matchingRow) ~= 1
        mismatchCount = mismatchCount + 1;
        continue;
    end

    originalValue = extractdata( ...
        originalLearnables.Value{i});

    updatedValue = extractdata( ...
        learnables.Value{matchingRow});

    if isequal(originalValue, updatedValue)
        preservedCount = preservedCount + 1;
    else
        mismatchCount = mismatchCount + 1;
    end

end

fprintf('Pretrained parameters preserved: %d\n', ...
    preservedCount);

fprintf('Parameter mismatches: %d\n', mismatchCount);

if mismatchCount > 0
    error('Some pretrained parameters were not preserved.');
end

%% 8. VERIFY CLASS ORDER AND NETWORK OUTPUT

fprintf('\nTarget class order:\n');

disp(poultryClasses);

fprintf('Network output names:\n');

disp(fineTuneNet.OutputNames);

%% 9. SAVE INITIALIZED NETWORK

savePath = fullfile(resultsFolder, ...
    'YAMNet_ThreeClass_Initialized.mat');

save(savePath, ...
    'fineTuneNet', ...
    'poultryClasses', ...
    'preservedCount', ...
    'mismatchCount');

fprintf('\n============================================\n');
fprintf(' STEP 6D COMPLETED\n');
fprintf('============================================\n');

fprintf('Three-class network initialized successfully.\n');
fprintf('Pretrained parameters preserved: %d\n', ...
    preservedCount);

fprintf('Saved file:\n%s\n', savePath);

fprintf('\nNo training has been performed yet.\n');
