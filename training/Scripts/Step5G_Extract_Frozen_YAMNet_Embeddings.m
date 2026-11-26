
%% STEP 5G: EXTRACT FROZEN YAMNET EMBEDDINGS
% CluckSense Smart Poultry
% YAMNet weights remain unchanged.
% One 1024-dimensional embedding per recording.

clc;
clear;
close all;

%% 1. PROJECT PATHS

projectFolder = pwd;

manifestFile = fullfile(projectFolder, ...
    "Dataset", "Dataset_Splits", ...
    "Dataset_Split_Manifest.csv");

resultsFolder = fullfile(projectFolder, "Results");

embeddingFolder = fullfile(projectFolder, ...
    "Features", "YAMNet_Embeddings");

if ~isfile(manifestFile)
    error("Dataset manifest not found.");
end

if ~isfolder(embeddingFolder)
    mkdir(embeddingFolder);
end

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

%% 2. LOAD MANIFEST USING ROBUST LINE PARSING

fileID = fopen(manifestFile,"r");

if fileID == -1
    error("Could not open dataset manifest.");
end

header = fgetl(fileID);

fileNames = strings(0,1);
classLabels = strings(0,1);
datasetSplits = strings(0,1);

while ~feof(fileID)

    line = fgetl(fileID);

    if isempty(line)
        continue;
    end

    parts = regexp(line, ...
        '^(.*),([^,]+),([^,]+)$', ...
        'tokens','once');

    if isempty(parts)
        fclose(fileID);
        error("Could not parse manifest line: %s",line);
    end

    fileNames(end+1,1) = string(parts{1});
    classLabels(end+1,1) = string(parts{2});
    datasetSplits(end+1,1) = string(parts{3});

end

fclose(fileID);

manifestTable = table(fileNames,classLabels,datasetSplits, ...
    'VariableNames', ...
    {'FileName','ClassLabel','DatasetSplit'});

numberOfFiles = height(manifestTable);

fprintf("Manifest loaded: %d recordings.\n",numberOfFiles);

%% 3. LOAD ORIGINAL PRETRAINED YAMNET

fprintf("\nLoading original pretrained YAMNet...\n");

[net,classNames] = audioPretrainedNetwork("yamnet");

fprintf("YAMNet loaded successfully.\n");

% The selected feature layer has 1024 output features.
embeddingLayer = "global_average_pooling2d";

embeddingDimension = 1024;

%% 4. PROCESSING SETTINGS

targetSampleRate = 16000;

minimumDuration = 1.5;

minimumSamples = round( ...
    minimumDuration * targetSampleRate);

windowBatchSize = 64;

%% 5. PREPARE EMBEDDING STORAGE

% One row per original recording.
% One column per embedding feature.

embeddings = zeros(numberOfFiles, ...
    embeddingDimension,"single");

processingStatus = strings(numberOfFiles,1);

numberOfWindows = zeros(numberOfFiles,1);

recordingDuration = zeros(numberOfFiles,1);

%% 6. PROCESS BY DATASET SPLIT

splitNames = ["Training","Validation","Testing"];

fprintf("\nStarting frozen YAMNet feature extraction...\n");

for s = 1:numel(splitNames)

    currentSplit = splitNames(s);

    splitIndices = find( ...
        manifestTable.DatasetSplit == currentSplit);

    fprintf("\n========================================\n");
    fprintf("PROCESSING %s SET\n",upper(currentSplit));
    fprintf("Recordings: %d\n",numel(splitIndices));
    fprintf("========================================\n");

    for j = 1:numel(splitIndices)

        idx = splitIndices(j);

        audioFile = manifestTable.FileName(idx);

        try

            %% Read audio

            [audioIn,fs] = audioread(audioFile);

            % Convert stereo to mono.
            if size(audioIn,2) > 1
                audioIn = mean(audioIn,2);
            end

            % Resample to 16 kHz.
            if fs ~= targetSampleRate

                audioIn = resample( ...
                    audioIn,targetSampleRate,fs);

                fs = targetSampleRate;

            end

            recordingDuration(idx) = numel(audioIn)/fs;

            % Pad short recordings in memory only.
            if numel(audioIn) < minimumSamples

                audioIn(end+1:minimumSamples,1) = 0;

            end

            %% YAMNet preprocessing

            spectrograms = yamnetPreprocess(audioIn,fs);

            totalWindows = size(spectrograms,4);

            if totalWindows == 0
                error("No YAMNet windows generated.");
            end

            numberOfWindows(idx) = totalWindows;

            %% Extract intermediate embeddings

            embeddingSum = zeros(1,embeddingDimension);

            for firstWindow = 1:windowBatchSize:totalWindows

                lastWindow = min( ...
                    firstWindow + windowBatchSize - 1, ...
                    totalWindows);

                batchSpectrograms = spectrograms(:,:,:, ...
                    firstWindow:lastWindow);

                dlInput = dlarray( ...
                    single(batchSpectrograms),"SSCB");

                batchOutput = predict(net,dlInput, ...
                    Outputs=embeddingLayer);

                batchOutput = gather(extractdata(batchOutput));

                % Convert [1 1 1024 batchSize]
                % into [batchSize 1024].
                batchMatrix = reshape(batchOutput, ...
                    embeddingDimension,[])';

                embeddingSum = embeddingSum + ...
                    sum(double(batchMatrix),1);

            end

            %% Average window embeddings

            recordingEmbedding = ...
                embeddingSum / totalWindows;

            embeddings(idx,:) = single(recordingEmbedding);

            processingStatus(idx) = "Success";

        catch ME

            processingStatus(idx) = "Failed";

            warning("Failed: %s\nReason: %s", ...
                audioFile,ME.message);

        end

        %% Progress display

        if mod(j,10) == 0 || j == numel(splitIndices)

            fprintf("%s: %d/%d processed.\n", ...
                currentSplit,j,numel(splitIndices));

        end

    end

end

%% 7. BUILD EMBEDDING METADATA

embeddingTable = table( ...
    manifestTable.FileName, ...
    manifestTable.ClassLabel, ...
    manifestTable.DatasetSplit, ...
    recordingDuration, ...
    numberOfWindows, ...
    processingStatus, ...
    'VariableNames',{ ...
    'FileName', ...
    'ClassLabel', ...
    'DatasetSplit', ...
    'DurationSeconds', ...
    'NumberOfWindows', ...
    'ProcessingStatus'});

%% 8. VERIFY EXTRACTION COMPLETENESS

successfulFiles = sum(processingStatus == "Success");

failedFiles = sum(processingStatus == "Failed");

fprintf("\n========================================\n");
fprintf("FROZEN YAMNET EMBEDDING SUMMARY\n");
fprintf("========================================\n");

fprintf("Total recordings: %d\n",numberOfFiles);
fprintf("Successful embeddings: %d\n",successfulFiles);
fprintf("Failed embeddings: %d\n",failedFiles);

fprintf("Embedding matrix dimensions:\n");
disp(size(embeddings));

for s = 1:numel(splitNames)

    currentSplit = splitNames(s);

    splitRows = manifestTable.DatasetSplit == currentSplit;

    successCount = sum( ...
        splitRows & processingStatus == "Success");

    fprintf("%s: %d successful embeddings\n", ...
        currentSplit,successCount);

end

%% 9. SAVE EMBEDDINGS AND METADATA

embeddingFile = fullfile(embeddingFolder, ...
    "Frozen_YAMNet_Embeddings.mat");

metadataFile = fullfile(embeddingFolder, ...
    "Frozen_YAMNet_Embedding_Metadata.csv");

save(embeddingFile, ...
    "embeddings", ...
    "embeddingTable", ...
    "manifestTable", ...
    "embeddingDimension", ...
    "embeddingLayer", ...
    "-v7.3");

writetable(embeddingTable,metadataFile);

%% 10. FINAL VERIFICATION

if successfulFiles == numberOfFiles && ...
        all(isfinite(embeddings(:)))

    fprintf("\nSUCCESS: All recordings have valid embeddings.\n");

else

    fprintf("\nWARNING: Extraction is incomplete.\n");

end

fprintf("\nSaved embeddings:\n%s\n",embeddingFile);

fprintf("\nSaved metadata:\n%s\n",metadataFile);

fprintf("\n========================================\n");
