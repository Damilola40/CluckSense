
clc;
clear;

%% Select processed dataset
processedPath = uigetdir(pwd, ...
    'Select Processed_Audio Folder');

%% Load processed recordings
ads = audioDatastore(processedPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', '.wav');

%% Display class counts
disp(countEachLabel(ads));

%% Read first processed recording
reset(ads);

[audio, info] = read(ads);

fprintf('New sampling rate: %d Hz\n', ...
    info.SampleRate);

fprintf('New duration: %.3f seconds\n', ...
    length(audio)/info.SampleRate);

fprintf('Number of samples: %d\n', ...
    length(audio));

%% Play processed audio
sound(audio, info.SampleRate);

disp('Verification completed.');
