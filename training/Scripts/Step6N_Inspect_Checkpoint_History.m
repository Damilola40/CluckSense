
%% STEP 6N: INSPECT CHECKPOINT HISTORY
clear; clc;

fprintf('\n============================================\n');
fprintf('STEP 6N: CHECKPOINT HISTORY INSPECTION\n');
fprintf('============================================\n\n');

%% 1. Load best checkpoint
file = fullfile(pwd,'Results', ...
    'YAMNet_FrozenHead_Best_Checkpoint.mat');

assert(isfile(file),'Checkpoint not found.');

C = load(file);

lossHistory = C.trainingLossHistory(:);
accuracyHistory = C.validationAccuracyHistory(:);

fprintf('Best epoch: %d\n',C.bestEpoch);
fprintf('Stored best validation accuracy: %.4f\n', ...
    C.bestValidationAccuracy);
fprintf('Best validation accuracy in percent: %.2f%%\n', ...
    100*C.bestValidationAccuracy);

%% 2. Inspect recorded history
fprintf('\nTRAINING HISTORY:\n');
fprintf('Epoch | Training Loss | Validation Accuracy\n');
fprintf('--------------------------------------------\n');

n = min(numel(lossHistory),numel(accuracyHistory));

for i = 1:n
    fprintf('%5d | %13.6f | %18.4f%%\n', ...
        i,lossHistory(i),100*accuracyHistory(i));
end

%% 3. Identify populated history entries
validRows = isfinite(lossHistory(1:n)) & ...
    isfinite(accuracyHistory(1:n)) & ...
    lossHistory(1:n) > 0;

fprintf('\nPopulated history entries: %d of %d\n', ...
    nnz(validRows),n);

%% 4. Inspect checkpoint Batch Normalization state
bestNet = C.bestNet;
state = bestNet.State;

fprintf('\nBATCH NORMALIZATION STATE SUMMARY:\n');

for i = 1:height(state)
    layerName = string(state.Layer(i));
    parameterName = string(state.Parameter(i));

    value = state.Value{i};

    fprintf('%-10s %-18s Size: %-18s\n', ...
        layerName,parameterName,mat2str(size(value)));
end

%% 5. Save inspection
historyInspection = table( ...
    (1:n)',lossHistory(1:n),accuracyHistory(1:n), ...
    validRows, ...
    'VariableNames', ...
    {'Epoch','TrainingLoss','ValidationAccuracy','Populated'});

save(fullfile(pwd,'Results', ...
    'Step6N_Checkpoint_History_Inspection.mat'), ...
    'historyInspection','validRows');

fprintf('\nInspection saved successfully.\n');

fprintf('\n============================================\n');
fprintf('STEP 6N INSPECTION COMPLETED\n');
fprintf('============================================\n');
