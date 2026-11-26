
%% STEP 6E: TEST THREE-CLASS YAMNET FORWARD PASS
% Purpose:
% Test whether the initialized three-class network
% accepts real YAMNet spectrogram input and produces
% three normalized class probabilities.

clear;
clc;

fprintf('\n============================================\n');
fprintf(' STEP 6E: TEST THREE-CLASS YAMNET\n');
fprintf('============================================\n\n');

%% 1. LOAD INITIALIZED NETWORK

projectFolder = ...
    'C:\Users\abdul\Desktop\CluckSense_Smart_Poultry';

modelPath = fullfile(projectFolder, ...
    'Results', 'YAMNet_ThreeClass_Initialized.mat');

data = load(modelPath);

fineTuneNet = data.fineTuneNet;
poultryClasses = data.poultryClasses;

% Use the established class order.
classNames = poultryClasses;

%% 2. SELECT ONE TRAINING RECORDING

manifestPath = fullfile(projectFolder, ...
    'Dataset', 'Dataset_Splits', ...
    'Dataset_Split_Manifest.csv');

fid = fopen(manifestPath, 'r');

if fid == -1
    error('Cannot open dataset manifest.');
end

fgetl(fid);

selectedFile = '';

while ~feof(fid)

    line = fgetl(fid);

    if ~ischar(line) || isempty(strtrim(line))
        continue;
    end

    parts = regexp(line, ...
        '^(.*),([^,]+),([^,]+)$', ...
        'tokens', 'once');

    if isempty(parts)
        continue;
    end

    if strcmp(parts{3}, 'Training')

        selectedFile = parts{1};
        actualClass = string(parts{2});
        break;

    end

end

fclose(fid);

if isempty(selectedFile)
    error('No training recording found.');
end

fprintf('Selected recording:\n%s\n', selectedFile);
fprintf('Actual class: %s\n', actualClass);

%% 3. LOAD AND PREPROCESS AUDIO

[audioIn, fs] = audioread(selectedFile);

if size(audioIn,2) > 1
    audioIn = mean(audioIn,2);
end

if fs ~= 16000
    audioIn = resample(audioIn, 16000, fs);
end

audioIn = single(audioIn);

minimumSamples = round(1.5 * 16000);

if numel(audioIn) < minimumSamples
    audioIn(end+1:minimumSamples,1) = 0;
end

spectrograms = yamnetPreprocess(audioIn, 16000);

fprintf('\nSpectrogram dimensions:\n');
disp(size(spectrograms));

%% 4. RUN NETWORK PREDICTION

fprintf('Running three-class network...\n');

scores = predict(fineTuneNet, spectrograms);

fprintf('Prediction completed.\n');

%% 5. INSPECT NETWORK OUTPUT

fprintf('\n============================================\n');
fprintf(' NETWORK OUTPUT DIAGNOSTICS\n');
fprintf('============================================\n');

fprintf('\nOutput variable class:\n');
disp(class(scores));

fprintf('\nOutput dimensions:\n');
disp(size(scores));

fprintf('\nOutput dimension labels:\n');

try
    disp(dims(scores));
catch
    fprintf('Dimension labels unavailable.\n');
end

scoreValues = extractdata(scores);

fprintf('\nRaw output matrix:\n');
disp(scoreValues);

fprintf('\nSum across rows:\n');
disp(sum(scoreValues,2));

fprintf('\nSum across columns:\n');
disp(sum(scoreValues,1));

fprintf('\nOutput layer details:\n');

outputLayer = getLayer(fineTuneNet, "softmax");
disp(outputLayer);

fprintf('\nClassification layer details:\n');

denseLayer = getLayer(fineTuneNet, "dense");
disp(denseLayer);

%% 6. EXTRACT AND VALIDATE CLASS PROBABILITIES

fprintf('\n============================================\n');
fprintf(' CLASS PROBABILITIES\n');
fprintf('============================================\n');

% Each row represents one audio window.
% Each column represents one poultry class.

if size(scoreValues,2) ~= numel(classNames)
    error('Output class count does not match poultry classes.');
end

% Extract the first audio window.
firstScores = scoreValues(1,:)';

fprintf('\nClass probabilities for the first window:\n');

for k = 1:numel(classNames)

    fprintf('%s: %.4f\n', ...
        string(classNames(k)), firstScores(k));

end

probabilitySum = sum(firstScores);

fprintf('\nProbability sum: %.4f\n', probabilitySum);

if any(~isfinite(scoreValues(:)))
    error('Network produced non-finite values.');
end

if any(scoreValues(:) < 0) || any(scoreValues(:) > 1)
    error('Network scores are outside the probability range.');
end

if any(abs(sum(scoreValues,2) - 1) > 1e-4)
    error('Probability normalization failed for one or more windows.');
end

fprintf('\nProbability normalization verified successfully.\n');

%% 7. COMPLETION MESSAGE

fprintf('\n============================================\n');
fprintf(' STEP 6E COMPLETED\n');
fprintf('============================================\n');

fprintf('Three-class network forward pass succeeded.\n');
fprintf('Three poultry classes verified.\n');
fprintf('Softmax probabilities verified.\n');
fprintf('No training or weight updates performed.\n');