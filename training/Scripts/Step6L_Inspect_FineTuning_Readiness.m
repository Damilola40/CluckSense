
%% STEP 6L: FINE-TUNING READINESS AUDIT
clear; clc;

fprintf('\n============================================\n');
fprintf('STEP 6L: FINE-TUNING READINESS AUDIT\n');
fprintf('============================================\n\n');

%% 1. Check required files
requiredFiles = {
    fullfile(pwd,'Results','YAMNet_FrozenHead_Trained.mat')
    fullfile(pwd,'Results','YAMNet_FrozenHead_Best_Checkpoint.mat')
    fullfile(pwd,'Results','FineTuning_Data_Inspection.mat')
    fullfile(pwd,'Dataset','Dataset_Splits', ...
        'Dataset_Split_Manifest.csv')
};

fprintf('REQUIRED FILE CHECK:\n\n');

for i = 1:numel(requiredFiles)
    fprintf('%d. %s\n',i,requiredFiles{i});
    assert(isfile(requiredFiles{i}), ...
        'Required file is missing.');
    fprintf('   STATUS: FOUND\n');
end

%% 2. Inspect frozen-head model
fprintf('\nFROZEN-HEAD MODEL INSPECTION:\n\n');

modelData = load(requiredFiles{1});

assert(isfield(modelData,'fineTuneNet'), ...
    'fineTuneNet variable is missing.');

fineTuneNet = modelData.fineTuneNet;

fprintf('Network type: %s\n',class(fineTuneNet));
fprintf('Network initialized: %d\n',fineTuneNet.Initialized);
fprintf('Learnable parameter groups: %d\n', ...
    height(fineTuneNet.Learnables));

%% 3. Inspect best checkpoint
fprintf('\nBEST CHECKPOINT INSPECTION:\n\n');

checkpointData = load(requiredFiles{2});

fprintf('Saved variables:\n');

checkpointVariables = fieldnames(checkpointData);

for i = 1:numel(checkpointVariables)
    fprintf('   %s\n',checkpointVariables{i});
end

%% 4. Inspect fine-tuning data inspection file
fprintf('\nFINE-TUNING DATA FILE INSPECTION:\n\n');

dataInspection = load(requiredFiles{3});

dataVariables = fieldnames(dataInspection);

for i = 1:numel(dataVariables)

    variableName = dataVariables{i};
    variableValue = dataInspection.(variableName);

    fprintf('\nVariable: %s\n',variableName);
    fprintf('Type: %s\n',class(variableValue));
    fprintf('Size: %s\n',mat2str(size(variableValue)));

    if istable(variableValue)
        fprintf('Table rows: %d\n',height(variableValue));
        fprintf('Table columns:\n');
        disp(variableValue.Properties.VariableNames);

    elseif isstruct(variableValue)
        fprintf('Structure fields:\n');
        disp(fieldnames(variableValue));
    end
end

%% 5. Inspect Batch Normalization state
fprintf('\nNETWORK STATE INSPECTION:\n\n');

networkState = fineTuneNet.State;

fprintf('State table size: %s\n', ...
    mat2str(size(networkState)));

if istable(networkState)
    disp(networkState(:, ...
        intersect({'Layer','Parameter'}, ...
        networkState.Properties.VariableNames)));
end

%% 6. Verify class definitions
fprintf('\nCLASS ORDER:\n');

classOrder = [
    "Background_Noise"
    "Healthy"
    "Respiratory_Abnormal"
];

disp(classOrder);

%% 7. Verify selective fine-tuning layers
fprintf('\nSELECTED FINE-TUNING LAYERS:\n');

learnables = fineTuneNet.Learnables;

selectedLayers = ["conv2d_13","dense"];

for i = 1:numel(selectedLayers)

    rows = find(string(learnables.Layer) == selectedLayers(i));

    fprintf('%s: %d parameter groups\n', ...
        selectedLayers(i),numel(rows));

    assert(~isempty(rows), ...
        'A selected fine-tuning layer is missing.');
end

fprintf('\n============================================\n');
fprintf('STEP 6L READINESS AUDIT COMPLETED\n');
fprintf('============================================\n');
