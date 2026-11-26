
%% STEP 6U-B: INSPECT EMBEDDING TABLE AND SVM INTERFACE

clear;
clc;

fprintf('\n');
fprintf('====================================================\n');
fprintf(' STEP 6U-B: EMBEDDING AND SVM INTERFACE INSPECTION\n');
fprintf('====================================================\n\n');

projectRoot = pwd;

embeddingPath = fullfile(projectRoot, ...
    'Features', 'YAMNet_Embeddings', ...
    'Frozen_YAMNet_Embeddings.mat');

svmPath = fullfile(projectRoot, ...
    'Results', 'Frozen_YAMNet_Baseline_Model.mat');

checkpointPath = fullfile(projectRoot, ...
    'Results', 'YAMNet_FrozenHead_Best_Checkpoint.mat');

%% 1. LOAD EMBEDDING DATA

E = load(embeddingPath);

fprintf('EMBEDDING TABLE VARIABLES\n');
disp(E.embeddingTable.Properties.VariableNames);

fprintf('\nFIRST FIVE EMBEDDING TABLE ROWS\n');
disp(E.embeddingTable(1:5, :));

fprintf('\nEMBEDDING MATRIX SIZE\n');
disp(size(E.embeddings));

%% 2. INSPECT MANIFEST TABLE

fprintf('\nMANIFEST TABLE VARIABLES\n');
disp(E.manifestTable.Properties.VariableNames);

fprintf('\nFIRST FIVE MANIFEST ROWS\n');
disp(E.manifestTable(1:5, :));

%% 3. LOAD SVM MODEL

S = load(svmPath);

fprintf('\nSVM MODEL CLASS\n');
disp(class(S.baselineModel));

fprintf('\nSVM CLASS ORDER\n');
disp(S.classOrder);

fprintf('\nFEATURE MEAN SIZE\n');
disp(size(S.featureMean));

fprintf('\nFEATURE STANDARD DEVIATION SIZE\n');
disp(size(S.featureStd));

%% 4. TEST SVM PREDICTION INTERFACE

fprintf('\nTESTING SVM PREDICTION INTERFACE\n');

Xtest = E.embeddings(1, :);

Xstandardized = ...
    (double(Xtest) - S.featureMean) ./ S.featureStd;

try

    predictedClass = predict( ...
        S.baselineModel, Xstandardized);

    fprintf('SVM prediction successful.\n');
    fprintf('Predicted class:\n');
    disp(predictedClass);

catch ME

    fprintf('SVM prediction failed.\n');
    fprintf('Error: %s\n', ME.message);

end

%% 5. INSPECT FROZEN-HEAD NETWORK OUTPUT

fprintf('\nTESTING FROZEN-HEAD NETWORK OUTPUT\n');

H = load(checkpointPath);

testInput = dlarray( ...
    rand(96, 64, 1, 2, 'single'), 'SSCB');

try

    scores = predict(H.bestNet, testInput);

    fprintf('Output class: %s\n', class(scores));
    fprintf('Output dimensions:\n');
    disp(size(scores));

    scoreValues = extractdata(scores);

    fprintf('Column probability sums:\n');
    disp(sum(scoreValues, 1));

catch ME

    fprintf('Network prediction failed.\n');
    fprintf('Error: %s\n', ME.message);

end

fprintf('\n====================================================\n');
fprintf(' STEP 6U-B INSPECTION COMPLETED\n');
fprintf('====================================================\n');

fprintf('No models were modified.\n');
fprintf('No training was performed.\n');
fprintf('No test recordings were evaluated.\n');
