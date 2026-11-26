
%% STEP 6B: PREPARE AND INSPECT YAMNET FINE-TUNING DATA
% Project: CluckSense Smart Poultry
% Purpose:
% 1. Read the existing dataset split manifest.
% 2. Verify every audio file.
% 3. Resample audio to 16 kHz.
% 4. Check YAMNet spectrogram preprocessing.
% 5. Record the number of YAMNet windows per recording.
%
% IMPORTANT:
% No model weights are modified in this step.
% Original audio files are not changed.

clear;
clc;

fprintf('\n============================================\n');
fprintf(' STEP 6B: FINE-TUNING DATA INSPECTION\n');
fprintf('============================================\n\n');

%% 1. PROJECT PATHS

projectFolder = ...
    'C:\Users\abdul\Desktop\CluckSense_Smart_Poultry';

manifestPath = fullfile(projectFolder, ...
    'Dataset', 'Dataset_Splits', ...
    'Dataset_Split_Manifest.csv');

resultsFolder = fullfile(projectFolder, 'Results');

if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

if ~exist(manifestPath, 'file')
    error('Dataset split manifest was not found.');
end

%% 2. LOAD YAMNET

fprintf('Loading pretrained YAMNet...\n');

[net, classNames] = audioPretrainedNetwork("yamnet");

fprintf('YAMNet loaded successfully.\n');
fprintf('Original AudioSet classes: %d\n\n', ...
    numel(classNames));

%% 3. READ MANIFEST

fprintf('Reading dataset manifest...\n');

fid = fopen(manifestPath, 'r');

if fid == -1
    error('Unable to open dataset manifest.');
end

headerLine = fgetl(fid);

fileNames = strings(0,1);
classLabels = strings(0,1);
datasetSplits = strings(0,1);

while ~feof(fid)

    line = fgetl(fid);

    if ~ischar(line) || isempty(strtrim(line))
        continue;
    end

    parts = regexp(line, ...
        '^(.*),([^,]+),([^,]+)$', ...
        'tokens', 'once');

    if isempty(parts)
        warning('Skipping unrecognized manifest line: %s', line);
        continue;
    end

    fileNames(end+1,1) = string(parts{1});
    classLabels(end+1,1) = string(parts{2});
    datasetSplits(end+1,1) = string(parts{3});

end

fclose(fid);

T = table(fileNames, classLabels, datasetSplits, ...
    'VariableNames', ...
    {'FileName','ClassLabel','DatasetSplit'});

fprintf('Manifest records loaded: %d\n\n', height(T));

%% 4. PREPARE INSPECTION VARIABLES

numberOfFiles = height(T);

fileExists = false(numberOfFiles,1);
audioReadable = false(numberOfFiles,1);
preprocessingSuccessful = false(numberOfFiles,1);

audioDuration = zeros(numberOfFiles,1);
yamnetWindowCount = zeros(numberOfFiles,1);

inputHeight = zeros(numberOfFiles,1);
inputWidth = zeros(numberOfFiles,1);
inputChannels = zeros(numberOfFiles,1);

status = strings(numberOfFiles,1);
errorMessage = strings(numberOfFiles,1);

targetSampleRate = 16000;
minimumDuration = 1.5;

%% 5. INSPECT EACH RECORDING

fprintf('Inspecting audio recordings...\n\n');

for i = 1:numberOfFiles

    currentFile = char(T.FileName(i));

    try

        % Confirm file existence
        if ~exist(currentFile, 'file')
            error('Audio file does not exist.');
        end

        fileExists(i) = true;

        % Read audio
        [audioIn, fs] = audioread(currentFile);

        audioReadable(i) = true;

        % Convert stereo to mono
        if size(audioIn,2) > 1
            audioIn = mean(audioIn,2);
        end

        % Resample to 16 kHz
        if fs ~= targetSampleRate
            audioIn = resample(audioIn, ...
                targetSampleRate, fs);
        end

        audioIn = single(audioIn);

        % Record original processed duration
        audioDuration(i) = numel(audioIn) / targetSampleRate;

        % Pad short recordings in memory only
        if numel(audioIn) < round(minimumDuration * targetSampleRate)

            requiredSamples = ...
                round(minimumDuration * targetSampleRate);

            audioIn(end+1:requiredSamples,1) = 0;

        end

        % Convert audio into YAMNet spectrogram input
        spectrograms = yamnetPreprocess( ...
            audioIn, targetSampleRate);

        % Check dimensions
        inputHeight(i) = size(spectrograms,1);
        inputWidth(i) = size(spectrograms,2);
        inputChannels(i) = size(spectrograms,3);

        yamnetWindowCount(i) = size(spectrograms,4);

        if inputHeight(i) ~= 96 || ...
           inputWidth(i) ~= 64 || ...
           inputChannels(i) ~= 1 || ...
           yamnetWindowCount(i) < 1

            error('Unexpected YAMNet input dimensions.');

        end

        preprocessingSuccessful(i) = true;
        status(i) = "SUCCESS";

    catch ME

        status(i) = "FAILED";
        errorMessage(i) = string(ME.message);

    end

    if mod(i,25) == 0 || i == numberOfFiles

        fprintf('Processed %d of %d recordings.\n', ...
            i, numberOfFiles);

    end

end

%% 6. CREATE INSPECTION TABLE

InspectionTable = table( ...
    T.FileName, ...
    T.ClassLabel, ...
    T.DatasetSplit, ...
    fileExists, ...
    audioReadable, ...
    audioDuration, ...
    yamnetWindowCount, ...
    inputHeight, ...
    inputWidth, ...
    inputChannels, ...
    preprocessingSuccessful, ...
    status, ...
    errorMessage, ...
    'VariableNames', { ...
    'FileName', ...
    'ClassLabel', ...
    'DatasetSplit', ...
    'FileExists', ...
    'AudioReadable', ...
    'DurationSeconds', ...
    'YAMNetWindows', ...
    'InputHeight', ...
    'InputWidth', ...
    'InputChannels', ...
    'PreprocessingSuccessful', ...
    'Status', ...
    'ErrorMessage'});

%% 7. DISPLAY OVERALL RESULTS

fprintf('\n============================================\n');
fprintf(' OVERALL INSPECTION RESULTS\n');
fprintf('============================================\n');

fprintf('Total recordings: %d\n', numberOfFiles);

fprintf('Files found: %d\n', sum(fileExists));

fprintf('Audio readable: %d\n', sum(audioReadable));

fprintf('Preprocessing successful: %d\n', ...
    sum(preprocessingSuccessful));

fprintf('Preprocessing failed: %d\n', ...
    sum(~preprocessingSuccessful));

fprintf('Total YAMNet windows: %d\n', ...
    sum(yamnetWindowCount));

fprintf('Average windows per recording: %.2f\n', ...
    mean(yamnetWindowCount(preprocessingSuccessful)));

fprintf('Maximum windows in one recording: %d\n', ...
    max(yamnetWindowCount));

fprintf('Minimum windows in one recording: %d\n', ...
    min(yamnetWindowCount(preprocessingSuccessful)));

%% 8. DISPLAY RESULTS BY DATASET SPLIT

fprintf('\n============================================\n');
fprintf(' RESULTS BY DATASET SPLIT\n');
fprintf('============================================\n');

splitNames = ["Training"; "Validation"; "Testing"];

for s = 1:numel(splitNames)

    selected = T.DatasetSplit == splitNames(s);

    fprintf('\n%s:\n', splitNames(s));

    fprintf('Recordings: %d\n', sum(selected));

    fprintf('Successful: %d\n', ...
        sum(preprocessingSuccessful(selected)));

    fprintf('Failed: %d\n', ...
        sum(~preprocessingSuccessful(selected)));

    fprintf('YAMNet windows: %d\n', ...
        sum(yamnetWindowCount(selected)));

end

%% 9. DISPLAY CLASS COUNTS

fprintf('\n============================================\n');
fprintf(' RECORDINGS BY CLASS\n');
fprintf('============================================\n');

classNamesList = unique(T.ClassLabel);

for c = 1:numel(classNamesList)

    fprintf('%s: %d recordings\n', ...
        classNamesList(c), ...
        sum(T.ClassLabel == classNamesList(c)));

end

%% 10. SAVE INSPECTION RESULTS

csvPath = fullfile(resultsFolder, ...
    'FineTuning_Data_Manifest.csv');

matPath = fullfile(resultsFolder, ...
    'FineTuning_Data_Inspection.mat');

writetable(InspectionTable, csvPath);

save(matPath, ...
    'InspectionTable', ...
    'T', ...
    'net', ...
    'classNames', ...
    'targetSampleRate', ...
    'minimumDuration');

fprintf('\n============================================\n');
fprintf(' STEP 6B COMPLETED\n');
fprintf('============================================\n');

fprintf('Saved inspection CSV:\n%s\n', csvPath);

fprintf('\nSaved inspection MAT file:\n%s\n', matPath);

if sum(preprocessingSuccessful) == numberOfFiles

    fprintf('\nALL RECORDINGS PASSED PREPROCESSING.\n');

else

    fprintf('\nWARNING: SOME RECORDINGS FAILED.\n');
    fprintf('Inspect the ErrorMessage column before proceeding.\n');

end
