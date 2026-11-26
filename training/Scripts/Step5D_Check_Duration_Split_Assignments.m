
%% PHASE 5.3C: CHECK DURATION DISTRIBUTION ACROSS SPLITS

clc;
clear;

projectFolder = pwd;

manifestFile = fullfile(projectFolder, ...
    'Dataset', 'Dataset_Splits', ...
    'Dataset_Split_Manifest.csv');

if ~isfile(manifestFile)
    error('Dataset split manifest not found.');
end


%% READ MANIFEST SAFELY

fid = fopen(manifestFile, 'r');

if fid == -1
    error('Unable to open dataset manifest.');
end

% Read and verify header
header = fgetl(fid);

if ~ischar(header)
    fclose(fid);
    error('Manifest is empty.');
end

fprintf('Manifest header: %s\n', header);

% Prepare columns
fileNames = strings(0,1);
classLabels = strings(0,1);
datasetSplits = strings(0,1);

lineNumber = 1;

while ~feof(fid)

    line = fgetl(fid);
    lineNumber = lineNumber + 1;

    if ~ischar(line) || isempty(strtrim(line))
        continue;
    end

    % Split from the RIGHT so commas in file paths are preserved
    parts = regexp(line, '^(.*),([^,]+),([^,]+)$', 'tokens', 'once');

    if isempty(parts)
        fclose(fid);
        error('Could not parse manifest line %d: %s', ...
            lineNumber, line);
    end

    filePath = string(strtrim(parts{1}));
    className = string(strtrim(parts{2}));
    splitName = string(strtrim(parts{3}));

    % Remove surrounding quotes if present
    filePath = strip(filePath, '"');

    fileNames(end+1,1) = filePath;
    classLabels(end+1,1) = className;
    datasetSplits(end+1,1) = splitName;

end

fclose(fid);

% Build a clean table
T = table(fileNames, classLabels, datasetSplits, ...
    'VariableNames', {'FileName', 'ClassLabel', 'DatasetSplit'});

fprintf('\nManifest records loaded: %d\n', height(T));
disp(T(1:min(5,height(T)), :));

numberOfFiles = height(T);
durations = zeros(numberOfFiles, 1);

fprintf('\nINSPECTING RECORDING DURATIONS...\n');

% Obtain duration of every recording
for k = 1:numberOfFiles

    audioPath = T.FileName(k);

    if ~isfile(audioPath)
        error('Audio file missing: %s', audioPath);
    end

    info = audioinfo(audioPath);
    durations(k) = info.Duration;

end

T.DurationSeconds = durations;

%% 1. COUNT RECORDINGS IN EACH SPLIT

fprintf('\n===== RECORDINGS PER DATASET SPLIT =====\n');

splitNames = ["Training", "Validation", "Testing"];

for k = 1:numel(splitNames)

    count = sum(T.DatasetSplit == splitNames(k));

    fprintf('%s: %d recordings\n', splitNames(k), count);

end

%% 2. DURATION SUMMARY BY SPLIT

fprintf('\n===== DURATION SUMMARY BY SPLIT =====\n');

for k = 1:numel(splitNames)

    selected = T.DurationSeconds(T.DatasetSplit == splitNames(k));

    fprintf('\n%s\n', splitNames(k));
    fprintf('Recordings: %d\n', numel(selected));
    fprintf('Minimum: %.3f seconds\n', min(selected));
    fprintf('Median: %.3f seconds\n', median(selected));
    fprintf('Maximum: %.3f seconds\n', max(selected));

end

%% 3. RECORDINGS SHORTER THAN ONE SECOND

fprintf('\n===== RECORDINGS SHORTER THAN ONE SECOND =====\n');

shortFiles = T(T.DurationSeconds < 1, :);

if isempty(shortFiles)

    fprintf('None found.\n');

else

    disp(shortFiles(:, ...
        {'FileName', 'ClassLabel', ...
         'DatasetSplit', 'DurationSeconds'}));

end

%% 4. RECORDINGS LONGER THAN TWO MINUTES

fprintf('\n===== RECORDINGS LONGER THAN 120 SECONDS =====\n');

longFiles = T(T.DurationSeconds > 120, :);

if isempty(longFiles)

    fprintf('None found.\n');

else

    disp(longFiles(:, ...
        {'FileName', 'ClassLabel', ...
         'DatasetSplit', 'DurationSeconds'}));

end

%% 5. SAVE INSPECTION RESULTS

outputFile = fullfile(projectFolder, ...
    'Results', 'Duration_Split_Inspection.mat');

if ~isfolder(fullfile(projectFolder, 'Results'))
    mkdir(fullfile(projectFolder, 'Results'));
end

save(outputFile, 'T', 'shortFiles', 'longFiles');

fprintf('\nInspection results saved to:\n%s\n', outputFile);

fprintf('\nPHASE 5.3C INSPECTION COMPLETE.\n');
