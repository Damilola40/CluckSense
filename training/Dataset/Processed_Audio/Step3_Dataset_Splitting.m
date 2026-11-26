
clc;
clear;
close all;

%% STEP 1: Select processed audio folder

processedPath = uigetdir(pwd, ...
    'Select Processed_Audio Folder');

if isequal(processedPath, 0)
    error('No folder selected.');
end

%% STEP 2: Load processed audio dataset

ads = audioDatastore(processedPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', '.wav');

files = string(ads.Files);
labels = ads.Labels;

totalFiles = numel(files);

fprintf('Total recordings: %d\n', totalFiles);

%% STEP 3: Set reproducible random seed

rng(42);

%% STEP 4: Prepare split assignments

split = strings(totalFiles,1);

classNames = categories(labels);

%% STEP 5: Split each class separately

for c = 1:numel(classNames)

    % Find recordings belonging to this class
    idx = find(labels == classNames{c});

    % Randomise their order
    idx = idx(randperm(numel(idx)));

    % Calculate allocation
    n = numel(idx);

    nTrain = floor(0.70*n);
    nValidation = floor(0.15*n);

    nTest = n - nTrain - nValidation;

    % Assign recordings
    trainIdx = idx(1:nTrain);

    validationIdx = idx( ...
        nTrain+1:nTrain+nValidation);

    testIdx = idx( ...
        nTrain+nValidation+1:end);

    split(trainIdx) = "Training";

    split(validationIdx) = "Validation";

    split(testIdx) = "Testing";

end

%% STEP 6: Create dataset manifest

datasetTable = table( ...
    files, string(labels), split, ...
    'VariableNames', ...
    {'FileName','ClassLabel','DatasetSplit'});

%% STEP 7: Display allocation

disp('DATASET SPLITTING RESULTS');

for c = 1:numel(classNames)

    className = string(classNames{c});

    fprintf('\n%s\n', className);

    for s = ["Training","Validation","Testing"]

        count = sum( ...
            string(labels) == className & split == s);

        fprintf('%s: %d\n', s, count);

    end

end

%% STEP 8: Save dataset information

outputFolder = fullfile( ...
    fileparts(processedPath), 'Dataset_Splits');

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

writetable(datasetTable, ...
    fullfile(outputFolder, ...
    'Dataset_Split_Manifest.csv'));

%% STEP 9: Save MATLAB variables

trainFiles = files(split == "Training");
validationFiles = files(split == "Validation");
testFiles = files(split == "Testing");

trainLabels = labels(split == "Training");
validationLabels = labels(split == "Validation");
testLabels = labels(split == "Testing");

save(fullfile(outputFolder, ...
    'Dataset_Splits.mat'), ...
    'trainFiles','validationFiles','testFiles', ...
    'trainLabels','validationLabels','testLabels');

%% STEP 10: Final summary

fprintf('\nTRAINING: %d\n', numel(trainFiles));

fprintf('VALIDATION: %d\n', ...
    numel(validationFiles));

fprintf('TESTING: %d\n', numel(testFiles));

fprintf('\nDataset splitting completed successfully.\n');
