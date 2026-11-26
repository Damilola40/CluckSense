
%% STEP 6U-A: INSPECT MODEL COMPARISON FILES
% Project: CluckSense Smart Poultry
% Purpose: Verify saved files before comparative validation.

clear;
clc;

fprintf('\n');
fprintf('====================================================\n');
fprintf(' STEP 6U-A: MODEL COMPARISON FILE INSPECTION\n');
fprintf('====================================================\n\n');

projectRoot = pwd;

filesToInspect = {
    fullfile(projectRoot, 'Results', ...
        'Frozen_YAMNet_Baseline_Model.mat');

    fullfile(projectRoot, 'Results', ...
        'YAMNet_FrozenHead_Best_Checkpoint.mat');

    fullfile(projectRoot, 'Results', ...
        'YAMNet_Selective_FineTuned_Best_Checkpoint.mat');

    fullfile(projectRoot, 'Results', ...
        'Frozen_YAMNet_Test_Predictions.csv');

    fullfile(projectRoot, 'Features', 'YAMNet_Embeddings', ...
        'Frozen_YAMNet_Embeddings.mat');

    fullfile(projectRoot, 'Results', ...
        'FineTuning_Data_Inspection.mat');
};

fileDescriptions = {
    'Frozen YAMNet SVM baseline model';
    'Frozen-head three-class YAMNet checkpoint';
    'Selectively fine-tuned YAMNet checkpoint';
    'Existing frozen YAMNet prediction CSV';
    'Frozen YAMNet embedding dataset';
    'Verified dataset manifest and inspection data'
};

%% 1. CHECK FILE EXISTENCE

fprintf('1. FILE EXISTENCE CHECK\n\n');

for i = 1:numel(filesToInspect)

    fprintf('%s\n', fileDescriptions{i});

    if isfile(filesToInspect{i})
        fprintf('STATUS: FOUND\n');
        fprintf('PATH: %s\n\n', filesToInspect{i});
    else
        fprintf('STATUS: NOT FOUND\n\n');
    end

end

%% 2. INSPECT MAT FILE VARIABLES

fprintf('\n2. MAT FILE VARIABLE INSPECTION\n\n');

matIndices = [1 2 3 5 6];

for i = matIndices

    currentFile = filesToInspect{i};

    fprintf('\n-----------------------------------------------\n');
    fprintf('%s\n', fileDescriptions{i});
    fprintf('-----------------------------------------------\n');

    if ~isfile(currentFile)
        fprintf('Skipped: file not found.\n');
        continue;
    end

    variableInformation = whos('-file', currentFile);

    for k = 1:numel(variableInformation)

        fprintf('Variable: %-35s | Class: %-15s | Size: ', ...
            variableInformation(k).name, ...
            variableInformation(k).class);

        fprintf('%s\n', mat2str(variableInformation(k).size));

    end

end

%% 3. INSPECT EXISTING PREDICTION CSV

fprintf('\n\n3. EXISTING PREDICTION CSV INSPECTION\n\n');

predictionFile = filesToInspect{4};

if isfile(predictionFile)

    predictionTable = readtable(predictionFile);

    fprintf('Rows: %d\n', height(predictionTable));
    fprintf('Columns:\n');

    disp(predictionTable.Properties.VariableNames);

    fprintf('First five rows:\n');
    disp(predictionTable(1:min(5,height(predictionTable)), :));

else

    fprintf('Prediction CSV not found.\n');

end

%% 4. INSPECT EMBEDDING DATASET STRUCTURE

fprintf('\n4. EMBEDDING DATASET INSPECTION\n\n');

embeddingFile = filesToInspect{5};

if isfile(embeddingFile)

    E = load(embeddingFile);

    fprintf('Embedding MAT variables:\n');
    disp(fieldnames(E));

    if isfield(E, 'embeddings')
        fprintf('Embeddings dimensions: %s\n', ...
            mat2str(size(E.embeddings)));
    end

    if isfield(E, 'embeddingMatrix')
        fprintf('Embedding matrix dimensions: %s\n', ...
            mat2str(size(E.embeddingMatrix)));
    end

else

    fprintf('Embedding dataset not found.\n');

end

%% 5. INSPECT FINE-TUNED CHECKPOINT

fprintf('\n5. SELECTIVELY FINE-TUNED CHECKPOINT INSPECTION\n\n');

fineTuneFile = filesToInspect{3};

if isfile(fineTuneFile)

    F = load(fineTuneFile);

    fprintf('Stored checkpoint fields:\n');
    disp(fieldnames(F));

    if isfield(F, 'bestEpoch')
        fprintf('Best epoch: %d\n', F.bestEpoch);
    end

    if isfield(F, 'bestValidationAccuracy')
        fprintf('Best validation accuracy: %.2f%%\n', ...
            100 * F.bestValidationAccuracy);
    end

    if isfield(F, 'validationPredictions')
        fprintf('Validation predictions are available.\n');
        fprintf('Prediction rows: %d\n', ...
            height(F.validationPredictions));
    else
        fprintf('Validation predictions are not stored in this checkpoint.\n');
    end

else

    fprintf('Fine-tuned checkpoint not found.\n');

end

%% 6. INSPECT FROZEN-HEAD CHECKPOINT

fprintf('\n6. FROZEN-HEAD CHECKPOINT INSPECTION\n\n');

frozenHeadFile = filesToInspect{2};

if isfile(frozenHeadFile)

    H = load(frozenHeadFile);

    fprintf('Stored checkpoint fields:\n');
    disp(fieldnames(H));

    if isfield(H, 'bestEpoch')
        fprintf('Best epoch: %d\n', H.bestEpoch);
    end

    if isfield(H, 'bestValidationAccuracy')
        fprintf('Best validation accuracy: %.2f%%\n', ...
            100 * H.bestValidationAccuracy);
    end

else

    fprintf('Frozen-head checkpoint not found.\n');

end

%% 7. INSPECT SVM BASELINE MODEL

fprintf('\n7. FROZEN YAMNET SVM BASELINE INSPECTION\n\n');

svmFile = filesToInspect{1};

if isfile(svmFile)

    S = load(svmFile);

    fprintf('Stored model fields:\n');
    disp(fieldnames(S));

    if isfield(S, 'classifier')
        fprintf('Classifier class: %s\n', class(S.classifier));
    end

    if isfield(S, 'poultryClasses')
        fprintf('Stored class order:\n');
        disp(S.poultryClasses);
    end

else

    fprintf('SVM baseline model not found.\n');

end

fprintf('\n====================================================\n');
fprintf(' STEP 6U-A INSPECTION COMPLETED\n');
fprintf('====================================================\n');

fprintf('No models were modified.\n');
fprintf('No training was performed.\n');
fprintf('No test recordings were accessed.\n');
