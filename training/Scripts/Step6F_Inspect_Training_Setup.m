
%% STEP 6F: INSPECT FINE-TUNING TRAINING SETUP
% Purpose:
% Verify the learnable parameters and training functions
% before training the new three-class classification head.
%
% IMPORTANT:
% This script does not train or modify the network.

clear;
clc;

fprintf('\n============================================\n');
fprintf(' STEP 6F: INSPECT TRAINING SETUP\n');
fprintf('============================================\n\n');

%% 1. LOAD INITIALIZED THREE-CLASS NETWORK

projectFolder = ...
    'C:\Users\abdul\Desktop\CluckSense_Smart_Poultry';

modelPath = fullfile(projectFolder, ...
    'Results', 'YAMNet_ThreeClass_Initialized.mat');

if ~isfile(modelPath)
    error('Initialized three-class model not found.');
end

data = load(modelPath);

fineTuneNet = data.fineTuneNet;
poultryClasses = data.poultryClasses;

fprintf('Initialized network loaded successfully.\n');

%% 2. DISPLAY CLASS ORDER

fprintf('\n============================================\n');
fprintf(' THREE POULTRY CLASSES\n');
fprintf('============================================\n');

disp(string(poultryClasses(:)));

if numel(poultryClasses) ~= 3
    error('Expected exactly three poultry classes.');
end

%% 3. INSPECT NETWORK LEARNABLE PARAMETERS

fprintf('\n============================================\n');
fprintf(' NETWORK LEARNABLE PARAMETERS\n');
fprintf('============================================\n');

learnables = fineTuneNet.Learnables;

fprintf('Total learnable parameter groups: %d\n', ...
    height(learnables));

fprintf('\nLearnable parameter names:\n');

disp(learnables(:, {'Layer', 'Parameter'}));

%% 4. IDENTIFY THE NEW DENSE CLASSIFICATION LAYER

fprintf('\n============================================\n');
fprintf(' NEW CLASSIFICATION LAYER\n');
fprintf('============================================\n');

denseRows = strcmp(string(learnables.Layer), "dense");

denseLearnables = learnables(denseRows,:);

disp(denseLearnables(:, {'Layer', 'Parameter'}));

if height(denseLearnables) ~= 2
    error('Expected two learnable parameters in dense layer.');
end

for k = 1:height(denseLearnables)

    parameterName = string(denseLearnables.Parameter(k));

    parameterValue = denseLearnables.Value{k};

    fprintf('\nParameter: %s\n', parameterName);

    fprintf('Dimensions: ');

    disp(size(parameterValue));

end

%% 5. VERIFY CLASSIFICATION LAYER DIMENSIONS

denseLayer = getLayer(fineTuneNet, "dense");

fprintf('\nDense layer input size: %d\n', ...
    denseLayer.InputSize);

fprintf('Dense layer output size: %d\n', ...
    denseLayer.OutputSize);

if denseLayer.InputSize ~= 1024
    error('Unexpected dense layer input size.');
end

if denseLayer.OutputSize ~= 3
    error('Dense layer must produce three class outputs.');
end

%% 6. COUNT FROZEN FEATURE-EXTRACTION PARAMETERS

fprintf('\n============================================\n');
fprintf(' FROZEN FEATURE EXTRACTOR\n');
fprintf('============================================\n');

frozenRows = ~denseRows;

fprintf('Pretrained parameter groups: %d\n', ...
    sum(frozenRows));

fprintf('New classification parameter groups: %d\n', ...
    sum(denseRows));

fprintf('Only the dense layer will be updated in this stage.\n');

%% 7. VERIFY REQUIRED TRAINING FUNCTIONS

fprintf('\n============================================\n');
fprintf(' TRAINING FUNCTION AVAILABILITY\n');
fprintf('============================================\n');

requiredFunctions = {
    'dlgradient'
    'dlfeval'
    'adamupdate'
    'yamnetPreprocess'
};

for k = 1:numel(requiredFunctions)

    functionName = requiredFunctions{k};

    functionLocation = which(functionName);

    fprintf('\n%s:\n', functionName);

    if isempty(functionLocation)

        fprintf('NOT FOUND\n');

        error('Required function %s is unavailable.', ...
            functionName);

    else

        fprintf('%s\n', functionLocation);

    end

end

%% 8. SAVE INSPECTION RESULTS

inspectionPath = fullfile(projectFolder, ...
    'Results', 'Step6F_Training_Setup_Inspection.mat');

save(inspectionPath, ...
    'poultryClasses', ...
    'denseLayer', ...
    'denseLearnables', ...
    'learnables');

fprintf('\n============================================\n');
fprintf(' STEP 6F INSPECTION COMPLETED\n');
fprintf('============================================\n');

fprintf('Network structure verified.\n');
fprintf('Three-class output verified.\n');
fprintf('Dense layer parameters identified.\n');
fprintf('Required training functions available.\n');

fprintf('\nNo training was performed.\n');
fprintf('No network weights were modified.\n');

fprintf('\nInspection saved to:\n%s\n', inspectionPath);
