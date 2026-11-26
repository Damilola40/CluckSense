clc;
clear;
close all;

%% STEP 1: Select dataset folder
datasetPath = uigetdir(pwd, 'Select Dataset Folder');

if isequal(datasetPath, 0)
    error('No dataset folder selected.');
end

%% STEP 2: Load WAV audio files
ads = audioDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', '.wav');

%% STEP 3: Check dataset
disp('POULTRY AUDIO DATASET');

classCounts = countEachLabel(ads);

disp(classCounts);

%% STEP 4: Display total recordings
totalFiles = numel(ads.Files);

fprintf('\nTotal audio files: %d\n', totalFiles);

%% STEP 5: Check for empty dataset
if totalFiles == 0
    error('No WAV files found. Check your folder structure.');
end

%% STEP 6: Display first recording
reset(ads);

[audio, info] = read(ads);

fs = info.SampleRate;

fprintf('\nOriginal sampling rate: %d Hz\n', fs);

fprintf('Audio duration: %.2f seconds\n', ...
    length(audio)/fs);

%% STEP 7: Play audio
sound(audio, fs);

%% STEP 8: Plot waveform
time = (0:length(audio)-1)/fs;

figure;

plot(time, audio);

xlabel('Time (seconds)');
ylabel('Amplitude');

title('Poultry Chick Audio Waveform');

grid on;

disp('Audio loading completed successfully.');
