
%% STEP 6A: INSPECT YAMNET ARCHITECTURE
% CluckSense Smart Poultry
% Inspect the pretrained network before fine-tuning.

clc;
clear;
close all;

%% 1. LOAD ORIGINAL PRETRAINED YAMNET

fprintf("Loading original pretrained YAMNet...\n");

[net, classNames] = audioPretrainedNetwork("yamnet");

fprintf("YAMNet loaded successfully.\n\n");

%% 2. GENERAL NETWORK INFORMATION

fprintf("========================================\n");
fprintf("YAMNET NETWORK INFORMATION\n");
fprintf("========================================\n");

fprintf("Number of layers: %d\n",numel(net.Layers));
fprintf("Number of original classes: %d\n",numel(classNames));

fprintf("\nInput names:\n");
disp(net.InputNames);

fprintf("Output names:\n");
disp(net.OutputNames);

%% 3. DISPLAY ALL NETWORK LAYERS

fprintf("\n========================================\n");
fprintf("COMPLETE YAMNET LAYER STRUCTURE\n");
fprintf("========================================\n");

for k = 1:numel(net.Layers)

    currentLayer = net.Layers(k);

    fprintf("%3d | %-35s | %s\n", ...
        k, ...
        currentLayer.Name, ...
        class(currentLayer));

end

%% 4. INSPECT FINAL CLASSIFICATION LAYERS

fprintf("\n========================================\n");
fprintf("FINAL CLASSIFICATION LAYERS\n");
fprintf("========================================\n");

numberOfLayers = numel(net.Layers);

for k = max(1,numberOfLayers-5):numberOfLayers

    fprintf("\nLayer %d:\n",k);
    disp(net.Layers(k));

end

%% 5. INSPECT LEARNABLE PARAMETERS

fprintf("\n========================================\n");
fprintf("LEARNABLE PARAMETERS\n");
fprintf("========================================\n");

learnableTable = net.Learnables;

fprintf("Total learnable parameter entries: %d\n", ...
    height(learnableTable));

for k = 1:height(learnableTable)

    parameterSize = size(learnableTable.Value{k});

    fprintf("%-30s | %-20s | ", ...
        string(learnableTable.Layer(k)), ...
        string(learnableTable.Parameter(k)));

    fprintf("%s\n",mat2str(parameterSize));

end

%% 6. INSPECT NETWORK CONNECTIONS

fprintf("\n========================================\n");
fprintf("FINAL NETWORK CONNECTIONS\n");
fprintf("========================================\n");

connectionTable = net.Connections;

disp(connectionTable(max(1,height(connectionTable)-10):end,:));

%% 7. SAVE INSPECTION INFORMATION

inspectionFile = fullfile(pwd, ...
    "Results", ...
    "YAMNet_Architecture_Inspection.mat");

save(inspectionFile, ...
    "net", ...
    "classNames", ...
    "learnableTable", ...
    "connectionTable", ...
    "-v7.3");

fprintf("\nArchitecture inspection saved:\n%s\n", ...
    inspectionFile);

fprintf("\nSTEP 6A COMPLETED.\n");
