
clc;
clear;

projectFolder = pwd;

manifestPath = fullfile(projectFolder, ...
    'Dataset', 'Dataset_Splits', ...
    'Dataset_Split_Manifest.csv');

fprintf('\nPROJECT FOLDER:\n%s\n', projectFolder);

fprintf('\nMANIFEST CHECK:\n');
if isfile(manifestPath)
    disp('Manifest found successfully.');
else
    error('Manifest not found. Check the project folder.');
end

fprintf('\nSEARCHING FOR PROCESSED AUDIO FOLDER:\n');

folders = dir(fullfile(projectFolder, '**', 'Processed_Audio'));

for k = 1:numel(folders)
    if folders(k).isdir
        fprintf('%s\n', fullfile(folders(k).folder, folders(k).name));
    end
end

fprintf('\nPATH CHECK COMPLETED.\n');
