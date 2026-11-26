
%% STEP 6O: INSPECT LEARNABLE PARAMETERS
clear; clc;

fprintf('\n============================================\n');
fprintf('STEP 6O: LEARNABLE PARAMETER INSPECTION\n');
fprintf('============================================\n\n');

%% 1. Load best checkpoint
checkpointFile = fullfile(pwd,'Results', ...
    'YAMNet_FrozenHead_Best_Checkpoint.mat');

assert(isfile(checkpointFile), ...
    'Best checkpoint file not found.');

C = load(checkpointFile);
net = C.bestNet;

fprintf('Best checkpoint loaded.\n');
fprintf('Network type: %s\n',class(net));

%% 2. Inspect learnable table
learnables = net.Learnables;

fprintf('\nLEARNABLE TABLE:\n');
fprintf('Rows: %d\n',height(learnables));
fprintf('Columns:\n');

disp(learnables.Properties.VariableNames);

%% 3. Inspect selected layers
selectedLayers = ["conv2d_13","dense"];

fprintf('\nSELECTED PARAMETER DETAILS:\n');

for k = 1:numel(selectedLayers)

    layerName = selectedLayers(k);

    fprintf('\nLayer: %s\n',layerName);

    rows = string(learnables.Layer) == layerName;

    selected = learnables(rows,:);

    for j = 1:height(selected)

        value = selected.Value{j};

        fprintf('Parameter: %s\n', ...
            string(selected.Parameter(j)));

        fprintf('Class: %s\n',class(value));
        fprintf('Size: %s\n',mat2str(size(value)));

        if isa(value,'dlarray')
            fprintf('Format: %s\n',dims(value));
        end
    end
end

%% 4. Verify all other layers can be identified
fprintf('\nFROZEN PARAMETER GROUPS:\n');

allLayers = string(learnables.Layer);

selectedMask = ...
    allLayers == "conv2d_13" | allLayers == "dense";

fprintf('Selected parameter groups: %d\n', ...
    nnz(selectedMask));

fprintf('Frozen parameter groups: %d\n', ...
    nnz(~selectedMask));

%% 5. Inspect network output with four observations
fprintf('\nNETWORK OUTPUT DIMENSION TEST:\n');

inputSize = [96 64 1 4];

dlX = dlarray(rand(single(inputSize)), ...
    'SSCB');

scores = predict(net,dlX);

fprintf('Input dimensions: %s\n',mat2str(size(dlX)));
fprintf('Output dimensions: %s\n',mat2str(size(scores)));

assert(size(scores,1) == 3, ...
    'Expected three output classes.');

assert(size(scores,2) == 4, ...
    'Expected four output observations.');

scoreValues = extractdata(scores);

fprintf('Output row sums:\n');
disp(sum(scoreValues,1));

assert(all(abs(sum(scoreValues,1)-1) < 1e-4), ...
    'Output probabilities do not sum to one.');

%% 6. Save inspection report
inspectionFile = fullfile(pwd,'Results', ...
    'Step6O_Learnable_Parameter_Inspection.mat');

save(inspectionFile, ...
    'learnables','selectedLayers','inputSize');

fprintf('\nInspection saved to:\n%s\n',inspectionFile);

fprintf('\n============================================\n');
fprintf('STEP 6O INSPECTION COMPLETED\n');
fprintf('============================================\n');
