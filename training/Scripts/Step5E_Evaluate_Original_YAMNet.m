
%% STEP 5E: EVALUATE ORIGINAL PRETRAINED YAMNET
% CluckSense Smart Poultry
% Original YAMNet is NOT fine-tuned or modified.
% Processes recordings independently, grouped by dataset split.

clc;
clear;
close all;

%% 1. PROJECT PATHS

projectFolder = pwd;

manifestFile = fullfile(projectFolder, ...
    "Dataset", "Dataset_Splits", ...
    "Dataset_Split_Manifest.csv");

resultsFolder = fullfile(projectFolder, "Results");

if ~isfile(manifestFile)
    error("Dataset manifest not found: %s", manifestFile);
end

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

%% 2. LOAD DATASET MANIFEST

fileID = fopen(manifestFile, "r");

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
        'tokens', 'once');

    if isempty(parts)
        fclose(fileID);
        error("Could not parse manifest line: %s", line);
    end

    fileNames(end+1,1) = string(parts{1});
    classLabels(end+1,1) = string(parts{2});
    datasetSplits(end+1,1) = string(parts{3});

end

fclose(fileID);

manifestTable = table(fileNames, classLabels, datasetSplits, ...
    'VariableNames', ...
    {'FileName','ClassLabel','DatasetSplit'});

fprintf("\nManifest loaded successfully.\n");
fprintf("Total recordings: %d\n", height(manifestTable));

%% 3. CHECK DATASET SPLITS

splitNames = ["Training","Validation","Testing"];

for k = 1:numel(splitNames)

    numberInSplit = sum(manifestTable.DatasetSplit == splitNames(k));

    fprintf("%s recordings: %d\n", ...
        splitNames(k), numberInSplit);

end

%% 4. LOAD ORIGINAL PRETRAINED YAMNET

fprintf("\nLoading original pretrained YAMNet...\n");

[net, classNames] = audioPretrainedNetwork("yamnet");

classNames = string(classNames(:));
numberOfClasses = numel(classNames);

fprintf("YAMNet loaded successfully.\n");
fprintf("Original AudioSet classes: %d\n", numberOfClasses);

%% 5. PROCESSING SETTINGS

targetSampleRate = 16000;

% YAMNet preprocessing requires sufficient audio duration.
minimumDuration = 1.5;
minimumSamples = round(minimumDuration * targetSampleRate);

% Process no more than 64 spectrogram windows per prediction call.
windowBatchSize = 64;

numberOfFiles = height(manifestTable);

predictedClass = strings(numberOfFiles,1);
predictedConfidence = zeros(numberOfFiles,1);
recordingDuration = zeros(numberOfFiles,1);
numberOfWindows = zeros(numberOfFiles,1);
processingStatus = strings(numberOfFiles,1);

% Store average 521-class scores for each recording.
recordingScores = zeros(numberOfFiles, numberOfClasses, "single");

%% 6. PROCESS RECORDINGS BY DATASET SPLIT

fprintf("\nStarting original YAMNet evaluation...\n");

for s = 1:numel(splitNames)

    currentSplit = splitNames(s);

    splitIndices = find(manifestTable.DatasetSplit == currentSplit);

    fprintf("\n========================================\n");
    fprintf("PROCESSING %s SET\n", upper(currentSplit));
    fprintf("Number of recordings: %d\n", numel(splitIndices));
    fprintf("========================================\n");

    for j = 1:numel(splitIndices)

        idx = splitIndices(j);

        audioFile = manifestTable.FileName(idx);

        try

            %% Read recording

            [audioIn, fs] = audioread(audioFile);

            % Convert stereo audio to mono.
            if size(audioIn,2) > 1
                audioIn = mean(audioIn,2);
            end

            % Resample if necessary.
            if fs ~= targetSampleRate
                audioIn = resample(audioIn, ...
                    targetSampleRate, fs);
                fs = targetSampleRate;
            end

            recordingDuration(idx) = numel(audioIn) / fs;

            % Pad short recordings without changing the source file.
            if numel(audioIn) < minimumSamples
                audioIn(end+1:minimumSamples,1) = 0;
            end

            %% YAMNet preprocessing

            spectrograms = yamnetPreprocess(audioIn, fs);

            totalWindows = size(spectrograms,4);

            numberOfWindows(idx) = totalWindows;

            if totalWindows == 0
                error("YAMNet preprocessing returned no windows.");
            end

            %% Predict in manageable batches

            scoreSum = zeros(1, numberOfClasses);

            for firstWindow = 1:windowBatchSize:totalWindows

                lastWindow = min( ...
                    firstWindow + windowBatchSize - 1, ...
                    totalWindows);

                batchSpectrograms = spectrograms(:,:,:, ...
                    firstWindow:lastWindow);

                batchScores = predict(net, batchSpectrograms);

                batchScores = gather(batchScores);

                % Ensure scores are arranged as:
                % number of windows x number of classes.
                if size(batchScores,2) == numberOfClasses

                    scoreSum = scoreSum + ...
                        sum(double(batchScores),1);

                elseif size(batchScores,1) == numberOfClasses

                    scoreSum = scoreSum + ...
                        sum(double(batchScores),2)';

                else

                    error("Unexpected YAMNet score dimensions.");

                end

            end

            %% Aggregate window scores per recording

            averageScores = scoreSum / totalWindows;

            recordingScores(idx,:) = single(averageScores);

            [confidence, classIndex] = max(averageScores);

            predictedClass(idx) = classNames(classIndex);

            predictedConfidence(idx) = confidence;

            processingStatus(idx) = "Success";

        catch ME

            processingStatus(idx) = "Failed";

            warning("Could not process %s: %s", ...
                audioFile, ME.message);

        end

        %% Progress display

        if mod(j,10) == 0 || j == numel(splitIndices)

            fprintf("%s: %d/%d recordings processed.\n", ...
                currentSplit, j, numel(splitIndices));

        end

    end

end

%% 7. CREATE RECORDING-LEVEL RESULTS

resultsTable = table( ...
    manifestTable.FileName, ...
    manifestTable.ClassLabel, ...
    manifestTable.DatasetSplit, ...
    recordingDuration, ...
    numberOfWindows, ...
    predictedClass, ...
    predictedConfidence, ...
    processingStatus, ...
    'VariableNames', { ...
    'FileName', ...
    'TrueClass', ...
    'DatasetSplit', ...
    'DurationSeconds', ...
    'NumberOfWindows', ...
    'PredictedAudioSetClass', ...
    'PredictionConfidence', ...
    'ProcessingStatus'});

%% 8. SAVE RESULTS

matFile = fullfile(resultsFolder, ...
    "Original_YAMNet_Evaluation.mat");

csvFile = fullfile(resultsFolder, ...
    "Original_YAMNet_Predictions.csv");

scoreFile = fullfile(resultsFolder, ...
    "Original_YAMNet_AverageScores.mat");

save(matFile, ...
    "resultsTable", ...
    "classNames", ...
    "recordingScores", ...
    "manifestTable", ...
    "-v7.3");

writetable(resultsTable, csvFile);

save(scoreFile, ...
    "recordingScores", ...
    "classNames", ...
    "fileNames", ...
    "-v7.3");

%% 9. DISPLAY PROCESSING SUMMARY

fprintf("\n========================================\n");
fprintf("ORIGINAL YAMNET EVALUATION SUMMARY\n");
fprintf("========================================\n");

successfulFiles = sum(processingStatus == "Success");
failedFiles = sum(processingStatus == "Failed");

fprintf("Total recordings: %d\n", numberOfFiles);
fprintf("Successfully processed: %d\n", successfulFiles);
fprintf("Failed recordings: %d\n", failedFiles);

for s = 1:numel(splitNames)

    currentSplit = splitNames(s);

    splitRows = manifestTable.DatasetSplit == currentSplit;

    successfulInSplit = sum( ...
        splitRows & processingStatus == "Success");

    fprintf("%s: %d successful recordings\n", ...
        currentSplit, successfulInSplit);

end

%% 10. DISPLAY MOST FREQUENT ORIGINAL YAMNET PREDICTIONS

validRows = processingStatus == "Success";

if any(validRows)

    validPredictions = predictedClass(validRows);

    [uniquePredictions,~,groupIndex] = unique(validPredictions);

    predictionCounts = accumarray(groupIndex,1);

    predictionSummary = table( ...
        uniquePredictions, predictionCounts, ...
        'VariableNames', ...
        {'AudioSetClass','RecordingCount'});

    predictionSummary = sortrows( ...
        predictionSummary, 'RecordingCount', 'descend');

    fprintf("\nMost frequent AudioSet predictions:\n");

    disp(predictionSummary( ...
        1:min(15,height(predictionSummary)),:));

    writetable(predictionSummary, ...
        fullfile(resultsFolder, ...
        "Original_YAMNet_Class_Summary.csv"));

end

fprintf("\n========================================\n");

fprintf("Original YAMNet evaluation completed.\n");

fprintf("Results saved in:\n%s\n", resultsFolder);

fprintf("========================================\n");
