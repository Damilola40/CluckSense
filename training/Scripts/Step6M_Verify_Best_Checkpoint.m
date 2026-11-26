
%% STEP 6M: VERIFY BEST FROZEN-HEAD CHECKPOINT
clear; clc;

fprintf('\n============================================\n');
fprintf('STEP 6M: BEST CHECKPOINT VERIFICATION\n');
fprintf('============================================\n\n');

%% 1. Load best checkpoint
checkpointFile = fullfile(pwd, ...
    'Results','YAMNet_FrozenHead_Best_Checkpoint.mat');

assert(isfile(checkpointFile), ...
    'Best checkpoint file not found.');

C = load(checkpointFile);

requiredVariables = {
    'bestNet'
    'bestEpoch'
    'bestValidationAccuracy'
    'poultryClasses'
    'trainingLossHistory'
    'validationAccuracyHistory'
};

for i = 1:numel(requiredVariables)
    assert(isfield(C,requiredVariables{i}), ...
        'Missing checkpoint variable: %s', ...
        requiredVariables{i});
end

bestNet = C.bestNet;

fprintf('Best checkpoint loaded successfully.\n');

%% 2. Display checkpoint performance
fprintf('\nCHECKPOINT PERFORMANCE:\n');
fprintf('Best epoch: %d\n',C.bestEpoch);
fprintf('Best validation accuracy: %.2f%%\n', ...
    C.bestValidationAccuracy);

fprintf('Training history length: %d\n', ...
    numel(C.trainingLossHistory));

fprintf('Validation history length: %d\n', ...
    numel(C.validationAccuracyHistory));

%% 3. Verify network
fprintf('\nNETWORK VERIFICATION:\n');
fprintf('Network type: %s\n',class(bestNet));
fprintf('Initialized: %d\n',bestNet.Initialized);
fprintf('Learnable groups: %d\n', ...
    height(bestNet.Learnables));

assert(bestNet.Initialized == 1, ...
    'Checkpoint network is not initialized.');

%% 4. Verify selected layers
learnables = bestNet.Learnables;

selectedLayers = ["conv2d_13","dense"];

fprintf('\nSELECTED LAYERS:\n');

for i = 1:numel(selectedLayers)

    rows = find(string(learnables.Layer) == selectedLayers(i));

    fprintf('%s: %d parameter groups\n', ...
        selectedLayers(i),numel(rows));

    assert(numel(rows) == 2, ...
        'Unexpected parameter group count.');
end

%% 5. Verify Batch Normalization state
fprintf('\nBATCH NORMALIZATION STATE:\n');

state = bestNet.State;

fprintf('State entries: %d\n',height(state));

assert(height(state) == 54, ...
    'Unexpected network state size.');

fprintf('All 54 state entries are present.\n');

%% 6. Verify class order
fprintf('\nCLASS ORDER:\n');
disp(string(C.poultryClasses));

expectedClasses = [
    "Background_Noise"
    "Healthy"
    "Respiratory_Abnormal"
];

assert(isequal(string(C.poultryClasses(:)),expectedClasses), ...
    'Checkpoint class order does not match expected order.');

%% 7. Save audit report
auditFile = fullfile(pwd, ...
    'Results','Step6M_Checkpoint_Verification.mat');

save(auditFile, ...
    'selectedLayers', ...
    'expectedClasses', ...
    'auditFile');

fprintf('\nAudit saved to:\n');
fprintf('Results\\Step6M_Checkpoint_Verification.mat\n');

fprintf('\n============================================\n');
fprintf('STEP 6M CHECKPOINT VERIFICATION COMPLETED\n');
fprintf('============================================\n');
