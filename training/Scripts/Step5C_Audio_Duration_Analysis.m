
%% PHASE 5.3A: AUDIO DURATION DISTRIBUTION ANALYSIS

clear;
clc;

projectFolder = ...
    'C:\Users\abdul\Desktop\CluckSense_Smart_Poultry';

classNames = {
    'Background_Noise'
    'Healthy'
    'Respiratory_Abnormal'
};

audioRoot = fullfile(projectFolder, ...
    'Dataset', 'Processed_Audio');

allDurations = [];
allClasses = strings(0,1);
allFiles = strings(0,1);

for c = 1:numel(classNames)

    folder = fullfile(audioRoot, classNames{c});
    files = dir(fullfile(folder, '*.wav'));

    durations = zeros(numel(files),1);

    for k = 1:numel(files)

        info = audioinfo(fullfile(folder, files(k).name));
        durations(k) = info.Duration;

        allDurations(end+1,1) = info.Duration;
        allClasses(end+1,1) = string(classNames{c});
        allFiles(end+1,1) = string(files(k).name);

    end

    sortedDurations = sort(durations);

    fprintf('\n====================================\n');
    fprintf('CLASS: %s\n', classNames{c});
    fprintf('====================================\n');

    fprintf('Recordings: %d\n', numel(durations));
    fprintf('Minimum: %.3f seconds\n', min(durations));
    fprintf('25th percentile: %.3f seconds\n', ...
        prctile(durations,25));
    fprintf('Median: %.3f seconds\n', median(durations));
    fprintf('75th percentile: %.3f seconds\n', ...
        prctile(durations,75));
    fprintf('90th percentile: %.3f seconds\n', ...
        prctile(durations,90));
    fprintf('Maximum: %.3f seconds\n', max(durations));

end

%% Overall duration statistics

fprintf('\n====================================\n');
fprintf('OVERALL DATASET\n');
fprintf('====================================\n');

fprintf('Total recordings: %d\n', numel(allDurations));
fprintf('Total audio hours: %.2f\n', sum(allDurations)/3600);
fprintf('Median duration: %.3f seconds\n', median(allDurations));

%% Display the 10 longest recordings

[~, order] = sort(allDurations, 'descend');

fprintf('\nTEN LONGEST RECORDINGS:\n');

for k = 1:min(10,numel(order))

    i = order(k);

    fprintf('%2d. %-25s %-25s %.2f seconds\n', ...
        k, allClasses(i), allFiles(i), allDurations(i));

end

%% Display recordings shorter than one second

shortIdx = find(allDurations < 1);

fprintf('\nRECORDINGS SHORTER THAN ONE SECOND: %d\n', ...
    numel(shortIdx));

for k = 1:numel(shortIdx)

    i = shortIdx(k);

    fprintf('%-25s %-25s %.3f seconds\n', ...
        allClasses(i), allFiles(i), allDurations(i));

end

fprintf('\nDURATION DISTRIBUTION ANALYSIS COMPLETE.\n');
