
clc;
clear;
close all;

%% STEP 1: Select original dataset folder

datasetPath = uigetdir(pwd, ...
    'Select Original Dataset Folder');

if isequal(datasetPath, 0)
    error('No dataset folder selected.');
end

%% STEP 2: Create processed audio folder

projectFolder = fileparts(datasetPath);

outputFolder = fullfile(projectFolder, ...
    'Processed_Audio');

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% STEP 3: Load original WAV recordings

ads = audioDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', '.wav');

totalFiles = numel(ads.Files);

fprintf('Total recordings: %d\n', totalFiles);

%% STEP 4: Set target sampling rate

targetFs = 16000;

%% STEP 5: Prepare processing information

fileNames = strings(totalFiles,1);
classLabels = strings(totalFiles,1);
originalRates = zeros(totalFiles,1);
newRates = zeros(totalFiles,1);
durations = zeros(totalFiles,1);
sampleCounts = zeros(totalFiles,1);

%% STEP 6: Process each recording

for k = 1:totalFiles

    % Get original file
    inputFile = ads.Files{k};

    % Get its class label
    label = string(ads.Labels(k));

    % Read audio
    [audio, originalFs] = audioread(inputFile);

    % Convert stereo to mono
    if size(audio,2) > 1
        audio = mean(audio,2);
    end

    % Resample to 16 kHz
    if originalFs ~= targetFs
        audio = resample(audio, targetFs, originalFs);
    end

    % Create class output folder
    classFolder = fullfile(outputFolder, char(label));

    if ~exist(classFolder, 'dir')
        mkdir(classFolder);
    end

    % Save processed WAV file
    [~, name, ext] = fileparts(inputFile);

    outputFile = fullfile(classFolder, ...
        [name ext]);

    audiowrite(outputFile, audio, targetFs);

    % Store processing information
    fileNames(k) = string([name ext]);
    classLabels(k) = label;
    originalRates(k) = originalFs;
    newRates(k) = targetFs;
    durations(k) = length(audio)/targetFs;
    sampleCounts(k) = length(audio);

    fprintf('Processed %d of %d files\n', ...
        k, totalFiles);

end

%% STEP 7: Create processing report

processingReport = table( ...
    fileNames, classLabels, ...
    originalRates, newRates, ...
    durations, sampleCounts);

disp('RESAMPLING SUMMARY');
disp(processingReport(1:min(10,totalFiles),:));

%% STEP 8: Save report

writetable(processingReport, ...
    fullfile(outputFolder, ...
    'Resampling_Report.csv'));

%% STEP 9: Display completion

fprintf('\nAll %d recordings processed.\n', ...
    totalFiles);

fprintf('New sampling rate: %d Hz\n', targetFs);

fprintf('Output folder: %s\n', outputFolder);

disp('RESAMPLING COMPLETED SUCCESSFULLY.');
