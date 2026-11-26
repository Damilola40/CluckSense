
%% STEP 6C: CREATE THREE-CLASS YAMNET ARCHITECTURE
% Project: CluckSense Smart Poultry
%
% Purpose:
% 1. Load the original pretrained YAMNet.
% 2. Replace its 521-class output layer with a 3-class layer.
% 3. Preserve the pretrained feature extractor.
% 4. Inspect and save the new architecture.
%
% IMPORTANT:
% This step does not train or modify the original saved baseline.

clear;
clc;

fprintf('\n============================================\n');
fprintf(' STEP 6C: CREATE THREE-CLASS YAMNET\n');
fprintf('============================================\n\n');

%% 1. DEFINE PROJECT PATHS

projectFolder = ...
    'C:\Users\abdul\Desktop\CluckSense_Smart_Poultry';

resultsFolder = fullfile(projectFolder, 'Results');

if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

%% 2. LOAD ORIGINAL PRETRAINED YAMNET

fprintf('Loading original pretrained YAMNet...\n');

[originalNet, audioSetClasses] = ...
    audioPretrainedNetwork("yamnet");

fprintf('Original YAMNet loaded.\n');
fprintf('Original output classes: %d\n', ...
    numel(audioSetClasses));

%% 3. CONFIRM ORIGINAL CLASSIFICATION HEAD

fprintf('\nInspecting original classification head...\n');

originalDenseLayer = getLayer(originalNet, "dense");

disp(originalDenseLayer);

fprintf('Original dense layer output size: %d\n', ...
    originalDenseLayer.OutputSize);

if originalDenseLayer.OutputSize ~= 521
    error('Unexpected original YAMNet classification head.');
end

%% 4. DEFINE POULTRY CLASS LABELS

poultryClasses = [
    "Background_Noise"
    "Healthy"
    "Respiratory_Abnormal"
];

numberOfClasses = numel(poultryClasses);

fprintf('\nTarget classification classes:\n');

disp(poultryClasses);

%% 5. REPLACE THE ORIGINAL 521-CLASS DENSE LAYER

fprintf('Replacing original 521-class dense layer...\n');

newClassificationLayer = fullyConnectedLayer( ...
    numberOfClasses, ...
    'Name', 'dense');

fineTuneNet = replaceLayer( ...
    originalNet, ...
    "dense", ...
    newClassificationLayer);

fprintf('Classification layer replaced successfully.\n');

%% 6. VERIFY THE NEW ARCHITECTURE

fprintf('\nInspecting new classification head...\n');

newDenseLayer = getLayer(fineTuneNet, "dense");

disp(newDenseLayer);

fprintf('New dense layer output size: %d\n', ...
    newDenseLayer.OutputSize);

if newDenseLayer.OutputSize ~= numberOfClasses
    error('New classification head does not have 3 outputs.');
end

%% 7. VERIFY NETWORK OUTPUT

fprintf('\nNetwork output names:\n');

disp(fineTuneNet.OutputNames);

fprintf('\nNetwork input names:\n');

disp(fineTuneNet.InputNames);

fprintf('\nNetwork initialized: %d\n', ...
    fineTuneNet.Initialized);

%% 8. DISPLAY UPDATED NETWORK SUMMARY

fprintf('\nUpdated network summary:\n');

summary(fineTuneNet);

%% 9. SAVE NEW ARCHITECTURE

architecturePath = fullfile( ...
    resultsFolder, ...
    'YAMNet_ThreeClass_Architecture.mat');

save(architecturePath, ...
    'fineTuneNet', ...
    'poultryClasses', ...
    'numberOfClasses', ...
    'audioSetClasses');

fprintf('\n============================================\n');
fprintf(' STEP 6C COMPLETED\n');
fprintf('============================================\n');

fprintf('Three-class output layer: %d classes\n', ...
    numberOfClasses);

fprintf('Saved architecture:\n%s\n', architecturePath);

fprintf('\nOriginal pretrained YAMNet remains unchanged.\n');
fprintf('No training has been performed yet.\n');
