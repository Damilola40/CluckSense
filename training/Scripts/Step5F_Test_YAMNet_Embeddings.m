
%% STEP 5F: TEST FROZEN YAMNET EMBEDDING EXTRACTION

clc;
clear;

%% 1. LOAD ORIGINAL YAMNET

[net, classNames] = audioPretrainedNetwork("yamnet");

fprintf("YAMNet loaded.\n");

%% 2. SELECT ONE RECORDING

testFile = fullfile(pwd, ...
    "Dataset", "Processed_Audio", ...
    "Respiratory_Abnormal", "1.wav");

if ~isfile(testFile)
    error("Test audio file not found: %s", testFile);
end

%% 3. READ AND PREPARE AUDIO

[audioIn, fs] = audioread(testFile);

if size(audioIn,2) > 1
    audioIn = mean(audioIn,2);
end

if fs ~= 16000
    audioIn = resample(audioIn,16000,fs);
    fs = 16000;
end

minimumSamples = round(1.5 * fs);

if numel(audioIn) < minimumSamples
    audioIn(end+1:minimumSamples,1) = 0;
end

fprintf("Audio duration: %.3f seconds\n", ...
    numel(audioIn)/fs);

%% 4. YAMNET PREPROCESSING

spectrograms = yamnetPreprocess(audioIn,fs);

fprintf("Spectrogram dimensions:\n");
disp(size(spectrograms));

%% 5. CONVERT INPUT TO FORMATTED DLARRAY

dlInput = dlarray(single(spectrograms),"SSCB");

%% 6. EXTRACT GLOBAL AVERAGE POOLING FEATURES

embeddingOutput = predict(net,dlInput, ...
    Outputs="global_average_pooling2d");

embeddingOutput = gather(extractdata(embeddingOutput));

fprintf("\nEmbedding output dimensions:\n");
disp(size(embeddingOutput));

%% 7. CONVERT TO ONE ROW PER AUDIO WINDOW

embeddingMatrix = reshape(embeddingOutput, ...
    [],size(embeddingOutput,4))';

fprintf("\nEmbedding matrix dimensions:\n");
disp(size(embeddingMatrix));

%% 8. AGGREGATE WINDOWS INTO ONE RECORDING EMBEDDING

recordingEmbedding = mean(embeddingMatrix,1);

fprintf("\nRecording embedding dimensions:\n");
disp(size(recordingEmbedding));

fprintf("\nEmbedding extraction test completed.\n");