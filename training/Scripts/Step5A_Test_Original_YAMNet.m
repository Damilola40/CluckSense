
%% PHASE 5.2: TEST ORIGINAL PRETRAINED YAMNET
% This test uses one respiratory-abnormality recording.
% The original 521-class YAMNet weights remain unchanged.

clear;
clc;

%% 1. Define project paths
projectFolder = 'C:\Users\abdul\Desktop\CluckSense_Smart_Poultry';

audioFolder = fullfile(projectFolder, ...
    'Dataset', 'Processed_Audio', 'Respiratory_Abnormal');

resultsFolder = fullfile(projectFolder, 'Results');

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

%% 2. Find one audio recording
audioFiles = dir(fullfile(audioFolder, '*.wav'));

if isempty(audioFiles)
    error('No WAV files found in: %s', audioFolder);
end

audioFile = fullfile(audioFolder, audioFiles(1).name);

fprintf('\nSelected recording:\n%s\n', audioFile);

%% 3. Load audio
[audioIn, fs] = audioread(audioFile);

% Convert stereo audio to mono if necessary
if size(audioIn, 2) > 1
    audioIn = mean(audioIn, 2);
end

fprintf('\nAudio sample rate: %d Hz\n', fs);
fprintf('Audio duration: %.3f seconds\n', ...
    numel(audioIn)/fs);

if fs ~= 16000
    error('Expected 16000 Hz audio. Check the processed audio file.');
end

%% 4. Load original pretrained YAMNet
[net, classNames] = audioPretrainedNetwork("yamnet");

fprintf('\nOriginal YAMNet loaded: %d classes\n', ...
    numel(classNames));

%% 5. Preprocess audio for YAMNet
yamnetFeatures = yamnetPreprocess(audioIn, fs);

fprintf('YAMNet preprocessing completed.\n');
fprintf('Feature dimensions: %s\n', ...
    mat2str(size(yamnetFeatures)));

%% 6. Run original YAMNet prediction
scores = predict(net, yamnetFeatures);

predictedLabels = scores2label(scores, classNames);

% Determine the most frequent prediction across audio frames
finalPrediction = mode(predictedLabels);

fprintf('\n====================================\n');
fprintf('ORIGINAL YAMNET TEST RESULT\n');
fprintf('====================================\n');

fprintf('Audio file: %s\n', audioFiles(1).name);
fprintf('Predicted AudioSet class: %s\n', ...
    string(finalPrediction));

%% 7. Save test results
save(fullfile(resultsFolder, ...
    'YAMNet_Original_Test.mat'), ...
    'audioFile', 'fs', 'yamnetFeatures', ...
    'scores', 'predictedLabels', 'finalPrediction');

fprintf('\nTest result saved successfully.\n');
fprintf('Location: %s\n', resultsFolder);

fprintf('\nPHASE 5.2 SINGLE-RECORDING TEST COMPLETE.\n');
