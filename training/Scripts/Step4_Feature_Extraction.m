
%% CLUCKSENSE - PHASE 4: FEATURE EXTRACTION
clc;
clear;
close all;

%% 1. PROJECT PATHS
projectFolder = pwd;

manifestPath = fullfile(projectFolder, ...
    'Dataset', 'Dataset_Splits', ...
    'Dataset_Split_Manifest.csv');

audioRoot = fullfile(projectFolder, ...
    'Dataset', 'Processed_Audio');

featureRoot = fullfile(projectFolder, 'Features');

assert(isfile(manifestPath), ...
    'Dataset manifest not found.');

assert(isfolder(audioRoot), ...
    'Processed_Audio folder not found.');

%% 2. READ DATASET MANIFEST

% Read the manifest line by line to preserve Windows file paths
fid = fopen(manifestPath, 'r');

assert(fid ~= -1, 'Cannot open dataset manifest.');

% Read and skip the header
headerLine = fgetl(fid);

% Read all remaining lines
rawLines = {};
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line) && ~isempty(strtrim(line))
        rawLines{end+1,1} = strtrim(line);
    end
end

fclose(fid);

% Prepare columns
numRows = numel(rawLines);
FileName = strings(numRows,1);
ClassLabel = strings(numRows,1);
DatasetSplit = strings(numRows,1);

% Extract the final three comma-separated fields.
% The Windows path is preserved as one complete path.
for i = 1:numRows

    parts = split(string(rawLines{i}), ',');

    assert(numel(parts) >= 3, ...
        'Invalid manifest row at line %d.', i+1);

    FileName(i) = strjoin(parts(1:end-2), ',');
    ClassLabel(i) = parts(end-1);
    DatasetSplit(i) = parts(end);

end

% Create the manifest table
manifest = table(FileName, ClassLabel, DatasetSplit);

fprintf('\nMANIFEST LOADED SUCCESSFULLY\n');
fprintf('Total recordings: %d\n', height(manifest));

fprintf('\nFIRST FIVE RECORDINGS:\n');
disp(manifest(1:min(5,height(manifest)), :));

fprintf('\nCLASS COUNTS:\n');
disp(groupcounts(manifest, 'ClassLabel'));

fprintf('\nSPLIT COUNTS:\n');
disp(groupcounts(manifest, 'DatasetSplit'));

% Confirm that all three columns exist
assert(width(manifest) == 3, ...
    'Manifest must contain exactly three columns.');


%% 3. FEATURE EXTRACTION SETTINGS
fs = 16000;

windowLength = 400;       % 25 ms
overlapLength = 240;      % 15 ms
fftLength = 512;

numCoeffs = 20;
numMelBands = 64;

window = hann(windowLength, 'periodic');

classes = ["Background_Noise", ...
           "Healthy", ...
           "Respiratory_Abnormal"];

splits = ["Training", "Validation", "Testing"];

%% 4. CREATE OUTPUT FOLDERS
featureTypes = ["Spectrograms", "MFCC", "LogMel"];

for f = 1:numel(featureTypes)
    for s = 1:numel(splits)
        for c = 1:numel(classes)

            outputFolder = fullfile(featureRoot, ...
                featureTypes(f), splits(s), classes(c));

            if ~isfolder(outputFolder)
                mkdir(outputFolder);
            end

        end
    end
end

%% 5. PROCESS EVERY RECORDING
processedCount = 0;
skippedCount = 0;

for k = 1:height(manifest)

    % Read information from the manifest
    originalPath = string(manifest.FileName(k));
    className = string(manifest.ClassLabel(k));
    splitName = lower(strtrim(string(manifest.DatasetSplit(k))));

    % Standardize split names
    if splitName == "training" || splitName == "train"
        splitName = "Training";
    elseif splitName == "validation" || splitName == "val"
        splitName = "Validation";
    elseif splitName == "testing" || splitName == "test"
        splitName = "Testing";
    else
        warning('Unknown split for row %d. Skipping.', k);
        skippedCount = skippedCount + 1;
        continue;
    end

    % Check class name
    if ~any(className == classes)
        warning('Unknown class at row %d. Skipping.', k);
        skippedCount = skippedCount + 1;
        continue;
    end

    % Locate corresponding processed audio file
    [~, baseName, ~] = fileparts(char(originalPath));

    audioPath = originalPath;

    if ~isfile(audioPath)
        warning('Processed audio not found: %s', audioPath);
        skippedCount = skippedCount + 1;
        continue;
    end

    % Read processed audio
    [audio, actualFs] = audioread(audioPath);

    % Convert stereo to mono if necessary
    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end

    % Confirm sampling rate
    if actualFs ~= fs
        warning('Unexpected sample rate in %s. Skipping.', audioPath);
        skippedCount = skippedCount + 1;
        continue;
    end

    %% A. SPECTROGRAM
    [S, F, T, P] = spectrogram(audio, ...
        window, overlapLength, fftLength, fs);

    spectrogramDB = 10*log10(P + eps);

    outputFolder = fullfile(featureRoot, ...
        'Spectrograms', splitName, className);

    outputName = sprintf('recording_%04d_%s.mat', k, baseName);

    save(fullfile(outputFolder, outputName), ...
        'S', 'F', 'T', 'P', 'spectrogramDB', 'fs');

    %% B. MFCC
    mfccFeatures = mfcc(audio, fs, ...
        'Window', window, ...
        'OverlapLength', overlapLength, ...
        'NumCoeffs', numCoeffs, ...
        'FFTLength', fftLength);

    outputFolder = fullfile(featureRoot, ...
        'MFCC', splitName, className);

    save(fullfile(outputFolder, outputName), ...
        'mfccFeatures', 'fs');

    %% C. LOG-MEL SPECTROGRAM
    melFeatures = melSpectrogram(audio, fs, ...
        'Window', window, ...
        'OverlapLength', overlapLength, ...
        'FFTLength', fftLength, ...
        'NumBands', numMelBands);

    logMelFeatures = log10(melFeatures + eps);

    outputFolder = fullfile(featureRoot, ...
        'LogMel', splitName, className);

    save(fullfile(outputFolder, outputName), ...
        'melFeatures', 'logMelFeatures', 'fs');

    processedCount = processedCount + 1;

    if mod(processedCount, 25) == 0
        fprintf('Processed %d recordings...\n', processedCount);
    end

end

%% 6. FINAL REPORT
fprintf('\n====================================\n');
fprintf('PHASE 4 FEATURE EXTRACTION COMPLETE\n');
fprintf('====================================\n');

fprintf('Manifest recordings: %d\n', height(manifest));
fprintf('Successfully processed: %d\n', processedCount);
fprintf('Skipped recordings: %d\n', skippedCount);

fprintf('\nFeature folders:\n');
disp(featureRoot);

fprintf('Expected files per feature family: %d\n', ...
    processedCount);

if processedCount == height(manifest) && skippedCount == 0
    disp('SUCCESS: All manifest recordings processed.');
else
    disp('CHECK: Some recordings were skipped. Review warnings.');
end
