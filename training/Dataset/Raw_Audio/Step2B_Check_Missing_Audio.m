
clc;
clear;

%% Select original dataset
originalPath = uigetdir(pwd, ...
    'Select Original Dataset');

%% Select processed dataset
processedPath = uigetdir(pwd, ...
    'Select Processed_Audio');

%% Count original WAV files
originalFiles = dir(fullfile( ...
    originalPath, '**', '*.wav'));

%% Count processed WAV files
processedFiles = dir(fullfile( ...
    processedPath, '**', '*.wav'));

fprintf('\nORIGINAL FILE COUNT: %d\n', ...
    length(originalFiles));

fprintf('PROCESSED FILE COUNT: %d\n', ...
    length(processedFiles));

%% Compare each class
classes = ["Background_Noise", ...
    "Healthy", ...
    "Respiratory_Abnormal"];

for i = 1:length(classes)

    className = classes(i);

    originalCount = 0;
    processedCount = 0;

    for k = 1:length(originalFiles)
        if contains(string(originalFiles(k).folder), ...
                className)
            originalCount = originalCount + 1;
        end
    end

    for k = 1:length(processedFiles)
        if contains(string(processedFiles(k).folder), ...
                className)
            processedCount = processedCount + 1;
        end
    end

    fprintf('\n%s\n', className);
    fprintf('Original: %d\n', originalCount);
    fprintf('Processed: %d\n', processedCount);
    fprintf('Difference: %d\n', ...
        originalCount - processedCount);

end
