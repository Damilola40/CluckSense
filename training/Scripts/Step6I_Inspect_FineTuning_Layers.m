
%% STEP 6I: INSPECT SELECTIVE FINE-TUNING LAYERS

clear;
clc;

fprintf('\n============================================\n');
fprintf('STEP 6I: SELECTIVE FINE-TUNING INSPECTION\n');
fprintf('============================================\n\n');

%% 1. Load the best frozen-head model

modelFile = fullfile(pwd, ...
    'Results','YAMNet_FrozenHead_Trained.mat');

assert(isfile(modelFile),'Trained frozen-head model not found.');

loadedData = load(modelFile);

fineTuneNet = loadedData.fineTuneNet;

fprintf('Best frozen-head model loaded successfully.\n');

%% 2. Inspect the final network layers

fprintf('\nFINAL NETWORK LAYERS:\n\n');

layers = fineTuneNet.Layers;

startLayer = max(1,numel(layers)-12);

for i = startLayer:numel(layers)

    layer = layers(i);

    fprintf('Layer %d: %s | %s\n', ...
        i,string(layer.Name),class(layer));

end

%% 3. Inspect learnable parameter groups

learnables = fineTuneNet.Learnables;

fprintf('\nLEARNABLE PARAMETER GROUPS:\n\n');

for i = 1:height(learnables)

    fprintf('%3d | %-22s | %-12s | ', ...
        i,string(learnables.Layer(i)), ...
        string(learnables.Parameter(i)));

    fprintf('%s\n', ...
        mat2str(size(learnables.Value{i})));

end

%% 4. Identify the final convolutional block

targetLayerNames = [
    "conv2d_13"
    "L132"
    "activation_26"
    "global_average_pooling2d"
    "dense"
];

fprintf('\nFINE-TUNING CANDIDATE LAYERS:\n\n');

for i = 1:numel(targetLayerNames)

    matchingRows = find( ...
        string(learnables.Layer) == targetLayerNames(i));

    fprintf('%s: %d learnable parameter groups\n', ...
        targetLayerNames(i),numel(matchingRows));

    for j = 1:numel(matchingRows)

        row = matchingRows(j);

        fprintf('   %s | %s\n', ...
            string(learnables.Parameter(row)), ...
            mat2str(size(learnables.Value{row})));

    end

end

%% 5. Verify the trained classification head

denseRows = find( ...
    string(learnables.Layer) == "dense");

assert(numel(denseRows) == 2, ...
    'Expected two dense-layer parameter groups.');

fprintf('\nThree-class classification head:\n');

for i = 1:numel(denseRows)

    row = denseRows(i);

    fprintf('%s: %s\n', ...
        string(learnables.Parameter(row)), ...
        mat2str(size(learnables.Value{row})));

end

%% 6. Save inspection results

inspectionFile = fullfile(pwd, ...
    'Results','Step6I_FineTuning_Layer_Inspection.mat');

save(inspectionFile, ...
    'targetLayerNames','learnables','-v7.3');

fprintf('\nInspection saved:\n');
fprintf('Results\\Step6I_FineTuning_Layer_Inspection.mat\n');

fprintf('\n============================================\n');
fprintf('STEP 6I INSPECTION COMPLETED\n');
fprintf('============================================\n');
