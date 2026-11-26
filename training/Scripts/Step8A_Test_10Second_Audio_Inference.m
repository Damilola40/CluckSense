
%% STEP 8A: TEST 10-SECOND AUDIO INFERENCE

clc;
clear;

fprintf('\nSTEP 8A: 10-SECOND AUDIO INFERENCE TEST\n');
fprintf('========================================\n');

%% 1. LOAD THE SAVED SVM MODEL

modelFile = fullfile(pwd, ...
    "Results", "Phase1_Team_Handover", ...
    "Phase1_Frozen_YAMNet_SVM_Model.mat");

if ~isfile(modelFile)
    error("Saved SVM model not found: %s", modelFile);
end

M = load(modelFile);

requiredFields = ["classifier", "featureMean", ...
    "featureStd", "classOrder", "embeddingDimension"];

for k = 1:numel(requiredFields)
    if ~isfield(M, requiredFields(k))
        error("Missing model field: %s", requiredFields(k));
    end
end

fprintf("Saved SVM model loaded successfully.\n");

%% 2. LOAD ORIGINAL YAMNET

[net, classNames] = audioPretrainedNetwork("yamnet");

fprintf("YAMNet loaded successfully.\n");
fprintf("AudioSet classes: %d\n", numel(classNames));

%% 3. SELECT A TRAINING RECORDING

testFile = fullfile(pwd, ...
    "Dataset", "Processed_Audio", ...
    "Respiratory_Abnormal", "65.wav");

if ~isfile(testFile)
    error("Test audio file not found: %s", testFile);
end

%% 4. READ AND PREPARE AUDIO

[audioIn, fs] = audioread(testFile);

if size(audioIn,2) > 1
    audioIn = mean(audioIn,2);
end

if fs ~= 16000
    audioIn = resample(audioIn,16000,fs);
    fs = 16000;
end

audioDuration = numel(audioIn)/fs;

fprintf("Audio duration: %.3f seconds\n", audioDuration);

if audioDuration < 10
    error("This recording is shorter than 10 seconds.");
end

%% 5. EXTRACT ONE 10-SECOND SEGMENT

segmentSamples = 10 * fs;

audioSegment = audioIn(1:segmentSamples);

fprintf("Inference segment: %.2f seconds\n", ...
    numel(audioSegment)/fs);

%% 6. YAMNET PREPROCESSING

spectrograms = yamnetPreprocess(audioSegment,fs);

fprintf("Spectrogram dimensions:\n");
disp(size(spectrograms));

%% 7. EXTRACT 1024-DIMENSIONAL EMBEDDINGS

dlInput = dlarray(single(spectrograms),"SSCB");

embeddingOutput = predict(net,dlInput, ...
    Outputs="global_average_pooling2d");

embeddingOutput = gather(extractdata(embeddingOutput));

embeddingMatrix = reshape(embeddingOutput, ...
    [],size(embeddingOutput,4))';

fprintf("Embedding matrix dimensions:\n");
disp(size(embeddingMatrix));

%% 8. AGGREGATE THE 10-SECOND SEGMENT

recordingEmbedding = mean(embeddingMatrix,1);

fprintf("Aggregated embedding dimensions:\n");
disp(size(recordingEmbedding));

%% 9. STANDARDISE FEATURES

if numel(recordingEmbedding) ~= M.embeddingDimension
    error("Embedding dimension does not match saved model.");
end

featureMean = double(M.featureMean);
featureStd = double(M.featureStd);

featureStd(featureStd == 0) = 1;

standardisedFeatures = ...
    (double(recordingEmbedding) - featureMean) ./ featureStd;

%% 10. CLASSIFY USING SAVED SVM

[predictedClass, svmScores] = ...
    predict(M.classifier,standardisedFeatures);

fprintf("\n10-SECOND CLASSIFICATION RESULT\n");
fprintf("================================\n");

fprintf("Predicted class: %s\n", string(predictedClass));

fprintf("\nCLASSIFICATION SCORES\n");

for k = 1:numel(M.classOrder)
    fprintf("%s: %.6f\n", ...
        string(M.classOrder(k)), svmScores(k));
end

fprintf("\nSTEP 8A COMPLETED SUCCESSFULLY.\n");
