%% CLUCKSENSE PHASE 1 SUBMISSION PACKAGER
% Creates a clean copy of the current CluckSense Phase 1 work.
% IMPORTANT:
%   1. Save this script in the main CluckSense_Smart_Poultry folder.
%   2. Run it from MATLAB.
%   3. It COPIES files. It does not move, rename, or delete your originals.
%   4. Review the generated file inventory before submitting the package.
%
% Expected source folders:
%   Dataset, Features, Models, Results
%
% The script also copies MATLAB .m and .mlx files from the project folder.

clc;
fprintf('\n============================================================\n');
fprintf('       CLUCKSENSE PHASE 1 SUBMISSION PACKAGER\n');
fprintf('============================================================\n\n');

%% 1. Find the main project folder
projectRoot = fileparts(mfilename('fullpath'));

requiredFolders = {'Dataset','Features','Models','Results'};
rootLooksCorrect = ~isempty(projectRoot) && isfolder(projectRoot);
if rootLooksCorrect
    for k = 1:numel(requiredFolders)
        if ~isfolder(fullfile(projectRoot, requiredFolders{k}))
            rootLooksCorrect = false;
            break;
        end
    end
end

if ~rootLooksCorrect
    fprintf('The script could not confirm the project folder automatically.\n');
    selectedFolder = uigetdir(pwd, 'Select the CluckSense_Smart_Poultry main folder');
    if isequal(selectedFolder, 0)
        fprintf('Operation cancelled. No files were changed.\n');
        return;
    end
    projectRoot = selectedFolder;
end

fprintf('Source project folder:\n%s\n\n', projectRoot);

%% 2. Create a new submission folder beside the original project
parentFolder = fileparts(projectRoot);
baseName = 'CluckSense_Phase1_Submission';
submissionRoot = fullfile(parentFolder, baseName);

if isfolder(submissionRoot)
    timeTag = datestr(now, 'yyyymmdd_HHMMSS');
    submissionRoot = fullfile(parentFolder, [baseName '_' timeTag]);
end

folderList = {
    '00_READ_ME_FIRST'
    '01_PROJECT_OVERVIEW'
    '02_DATASET_AND_PREPROCESSING'
    '03_SIGNAL_PROCESSING_AND_FEATURES'
    '04_ORIGINAL_YAMNET_BASELINE'
    '05_FROZEN_YAMNET_SVM'
    '06_YAMNET_FINE_TUNING'
    '07_RESULTS_AND_MODEL_COMPARISONS'
    '08_COMPLETE_MATLAB_SOURCE_CODE'
    '09_MODELS_AND_CHECKPOINTS'
    '10_FULL_DATA_AND_FEATURE_ARCHIVE'
    '11_NEXT_STAGE_CNN_ESP32'
};

[ok, msg] = mkdir(submissionRoot);
if ~ok
    error('Could not create submission folder: %s', msg);
end

for k = 1:numel(folderList)
    mkdir(fullfile(submissionRoot, folderList{k}));
end

fprintf('Created submission folder:\n%s\n\n', submissionRoot);

%% 3. Set up the file inventory
% Columns: source path, submission path, category, copy status, file size (bytes)
records = cell(0,5);
copyCount = 0;
failCount = 0;

%% 4. Copy the main data, feature and model folders
% Full folders are included so the work can be reviewed and reproduced.
% These copies may make the submission package large.

sourceDataset = fullfile(projectRoot, 'Dataset');
if isfolder(sourceDataset)
    destination = fullfile(submissionRoot, '10_FULL_DATA_AND_FEATURE_ARCHIVE', 'Dataset');
    [records, nOK, nFail] = copyTree(sourceDataset, destination, 'Complete Dataset', records);
    copyCount = copyCount + nOK;
    failCount = failCount + nFail;
else
    fprintf('WARNING: Dataset folder was not found.\n');
end

sourceFeatures = fullfile(projectRoot, 'Features');
if isfolder(sourceFeatures)
    destination = fullfile(submissionRoot, '10_FULL_DATA_AND_FEATURE_ARCHIVE', 'Features');
    [records, nOK, nFail] = copyTree(sourceFeatures, destination, 'Complete Feature Archive', records);
    copyCount = copyCount + nOK;
    failCount = failCount + nFail;
else
    fprintf('WARNING: Features folder was not found.\n');
end

sourceModels = fullfile(projectRoot, 'Models');
if isfolder(sourceModels)
    destination = fullfile(submissionRoot, '09_MODELS_AND_CHECKPOINTS', 'Models');
    [records, nOK, nFail] = copyTree(sourceModels, destination, 'Saved Models Folder', records);
    copyCount = copyCount + nOK;
    failCount = failCount + nFail;
else
    fprintf('NOTE: Models folder was not found.\n');
end

%% 5. Copy and classify every file in Results
sourceResults = fullfile(projectRoot, 'Results');
if isfolder(sourceResults)
    resultFiles = dir(fullfile(sourceResults, '**', '*'));
    resultFiles = resultFiles(~[resultFiles.isdir]);

    fprintf('Organising %d file(s) from Results...\n', numel(resultFiles));
    for k = 1:numel(resultFiles)
        sourceFile = fullfile(resultFiles(k).folder, resultFiles(k).name);
        categoryFolder = classifyResultFile(resultFiles(k).name);
        categoryPath = fullfile(submissionRoot, categoryFolder, 'From_Results');

        % Preserve any subfolder structure that exists under Results.
        relativePath = sourceFile(numel(sourceResults)+2:end);
        destinationFile = fullfile(categoryPath, relativePath);
        [records, copied] = copyOneFile(sourceFile, destinationFile, categoryFolder, records);
        if copied
            copyCount = copyCount + 1;
        else
            failCount = failCount + 1;
        end
    end
else
    fprintf('WARNING: Results folder was not found.\n');
end

%% 6. Copy MATLAB source code (.m and .mlx), preserving subfolders
fprintf('Collecting MATLAB source files...\n');
codeDestination = fullfile(submissionRoot, '08_COMPLETE_MATLAB_SOURCE_CODE');
codeExtensions = {'*.m','*.mlx'};

for e = 1:numel(codeExtensions)
    codeFiles = dir(fullfile(projectRoot, '**', codeExtensions{e}));
    for k = 1:numel(codeFiles)
        sourceFile = fullfile(codeFiles(k).folder, codeFiles(k).name);

        % Safety guard: do not package files from the output folder if it
        % happens to be created inside the selected project directory.
        if startsWith(sourceFile, [submissionRoot filesep], 'IgnoreCase', true)
            continue;
        end

        relativePath = sourceFile(numel(projectRoot)+2:end);
        destinationFile = fullfile(codeDestination, relativePath);
        [records, copied] = copyOneFile(sourceFile, destinationFile, 'MATLAB Source Code', records);
        if copied
            copyCount = copyCount + 1;
        else
            failCount = failCount + 1;
        end
    end
end

%% 7. Copy any MATLAB data files stored directly in the project root
rootMatFiles = dir(fullfile(projectRoot, '*.mat'));
if ~isempty(rootMatFiles)
    extraMatFolder = fullfile(submissionRoot, '09_MODELS_AND_CHECKPOINTS', 'Root_Level_MAT_Files');
    for k = 1:numel(rootMatFiles)
        sourceFile = fullfile(rootMatFiles(k).folder, rootMatFiles(k).name);
        destinationFile = fullfile(extraMatFolder, rootMatFiles(k).name);
        [records, copied] = copyOneFile(sourceFile, destinationFile, 'Root MATLAB Data File - Review Contents', records);
        if copied
            copyCount = copyCount + 1;
        else
            failCount = failCount + 1;
        end
    end
end

%% 8. Create a human-readable project overview and status note
readmePath = fullfile(submissionRoot, '00_READ_ME_FIRST', 'README_FIRST.txt');
writeTextFile(readmePath, {
    'CLUCKSENSE - PHASE 1 WORK PACKAGE'
    'Team: PHAL (Poultry Health Alert)'
    ''
    'PURPOSE'
    'This folder is an organised COPY of the CluckSense MATLAB project files.'
    'The original project folder has not been moved or modified by this script.'
    ''
    'PROJECT WORK RECORDED SO FAR'
    '- Dataset of 345 poultry audio recordings.'
    '- Audio preprocessing at 16 kHz.'
    '- Train/validation/test split: 240 / 50 / 55 recordings.'
    '- Spectrogram, MFCC and Log-Mel feature extraction.'
    '- Original YAMNet evaluation.'
    '- Frozen YAMNet embeddings and SVM classification.'
    '- Selective YAMNet fine-tuning experiment.'
    ''
    'RECORDED FROZEN YAMNET + SVM RESULTS'
    '- Validation accuracy: 84.00%.'
    '- Test accuracy: 83.64% (46 correct out of 55 test recordings).'
    '- Test macro-F1: 82.74%.'
    ''
    'FINE-TUNING STATUS'
    '- Best recorded validation accuracy: 78% at epoch 1.'
    '- Held-out test performance must be confirmed separately.'
    ''
    'IMPORTANT SCIENTIFIC SCOPE'
    'The three current labels are Background_Noise, Healthy and Respiratory_Abnormal.'
    'These results describe three-class acoustic classification. They do not yet'
    'demonstrate validated heat-stress detection in live poultry.'
    ''
    'NEXT PHASE 1 MILESTONE'
    'Build and evaluate a lightweight custom CNN, then assess feasibility for ESP32 deployment.'
    'The CNN/ESP32 work is a pending milestone and must not be described as completed.'
    ''
    'BEFORE SUBMISSION'
    '1. Review Submission_File_Inventory.csv.'
    '2. Open key MATLAB scripts and confirm their dependencies and paths.'
    '3. Open saved MAT files and check that the required model variables are present.'
    '4. Confirm that all reported figures match the final MATLAB outputs.'
    '5. Remove private, temporary or unrelated files if any were copied.'
    '6. Check the hackathon file-size and data-sharing rules before uploading raw audio.'
    ''
    ['Created: ' datestr(now, 'dd-mmm-yyyy HH:MM:SS')]
    ['Original project: ' projectRoot]
    ['Submission package: ' submissionRoot]
});

statusPath = fullfile(submissionRoot, '01_PROJECT_OVERVIEW', 'PHASE1_STATUS_AND_SCOPE.txt');
writeTextFile(statusPath, {
    'PHASE 1 - CURRENT STATUS'
    ''
    'COMPLETED / RECORDED'
    'Dataset preparation, 16 kHz preprocessing, feature extraction, original YAMNet'
    'evaluation, frozen YAMNet + SVM baseline, and a selective fine-tuning experiment.'
    ''
    'RESULTS CURRENTLY RECORDED'
    'Frozen YAMNet + SVM: 84.00% validation accuracy, 83.64% test accuracy,'
    'and 82.74% test macro-F1.'
    'Selective fine-tuned YAMNet: best validation accuracy 78% at epoch 1.'
    'The fine-tuned checkpoint still requires a separate held-out test evaluation.'
    ''
    'REMAINING FINAL PHASE 1 MILESTONE'
    'Develop a lightweight custom CNN for the three existing acoustic classes.'
    'Evaluate the model and assess model size, RAM, inference time and deployment route'
    'for the intended ESP32 variant.'
    ''
    'CAUTION'
    'Do not describe the current three-class acoustic results as heat-stress detection'
    'accuracy. Heat-stress-specific recordings and independent environmental ground truth'
    'are still needed for that claim.'
});

pendingPath = fullfile(submissionRoot, '11_NEXT_STAGE_CNN_ESP32', 'STATUS_NOT_YET_COMPLETED.txt');
writeTextFile(pendingPath, {
    'NEXT DEVELOPMENT MILESTONE - NOT YET COMPLETED'
    ''
    'Build a compact CNN for Background_Noise, Healthy and Respiratory_Abnormal.'
    'Choose and standardise fixed-duration audio windows and input features.'
    'Train using the existing training split and select settings using validation data.'
    'Evaluate once on the untouched test split.'
    'Measure model size, peak RAM use and inference time.'
    'Investigate quantisation and ESP32-compatible deployment.'
    ''
    'This folder is a planning note only. It is not evidence that CNN deployment is complete.'
});

%% 9. Write file inventory
if isempty(records)
    inventory = cell2table(cell(0,5), 'VariableNames', ...
        {'SourcePath','SubmissionPath','Category','CopyStatus','SizeBytes'});
else
    inventory = cell2table(records, 'VariableNames', ...
        {'SourcePath','SubmissionPath','Category','CopyStatus','SizeBytes'});
end

inventoryPath = fullfile(submissionRoot, 'Submission_File_Inventory.csv');
writetable(inventory, inventoryPath);

%% 10. Write a packaging log
logPath = fullfile(submissionRoot, 'Packaging_Log.txt');
writeTextFile(logPath, {
    'CLUCKSENSE PHASE 1 PACKAGING LOG'
    ['Source: ' projectRoot]
    ['Destination: ' submissionRoot]
    ['Files copied successfully: ' num2str(copyCount)]
    ['Files with copy errors: ' num2str(failCount)]
    ['Date: ' datestr(now, 'dd-mmm-yyyy HH:MM:SS')]
    ''
    'This process copied files and did not delete or move the originals.'
    'Review Submission_File_Inventory.csv and inspect the package before submission.'
});

%% 11. Display completion summary
fprintf('\n============================================================\n');
fprintf('             PACKAGING PROCESS FINISHED\n');
fprintf('============================================================\n');
fprintf('Files copied successfully : %d\n', copyCount);
fprintf('Files with copy errors    : %d\n', failCount);
fprintf('\nSubmission package created at:\n%s\n', submissionRoot);
fprintf('\nOpen the new folder and review:\n');
fprintf('  1. 00_READ_ME_FIRST\\README_FIRST.txt\n');
fprintf('  2. Submission_File_Inventory.csv\n');
fprintf('  3. Packaging_Log.txt\n');
fprintf('\nIMPORTANT: The CNN/ESP32 milestone is marked as pending.\n');
fprintf('============================================================\n\n');

if failCount > 0
    warning('Some files could not be copied. Check Packaging_Log.txt and the inventory.');
end

%% LOCAL FUNCTIONS
function [records, copiedCount, failedCount] = copyTree(sourceFolder, destinationFolder, category, records)
    copiedCount = 0;
    failedCount = 0;
    if ~isfolder(sourceFolder)
        return;
    end
    allItems = dir(fullfile(sourceFolder, '**', '*'));
    allItems = allItems(~[allItems.isdir]);
    for i = 1:numel(allItems)
        sourceFile = fullfile(allItems(i).folder, allItems(i).name);
        relativePath = sourceFile(numel(sourceFolder)+2:end);
        destinationFile = fullfile(destinationFolder, relativePath);
        [records, copied] = copyOneFile(sourceFile, destinationFile, category, records);
        if copied
            copiedCount = copiedCount + 1;
        else
            failedCount = failedCount + 1;
        end
    end
end

function [records, copied] = copyOneFile(sourceFile, destinationFile, category, records)
    copied = false;
    fileInfo = dir(sourceFile);
    if isempty(fileInfo)
        records(end+1,:) = {sourceFile, destinationFile, category, 'SOURCE_NOT_FOUND', 0}; %#ok<AGROW>
        return;
    end

    parentFolder = fileparts(destinationFile);
    if ~isfolder(parentFolder)
        mkdir(parentFolder);
    end

    try
        [status, message] = copyfile(sourceFile, destinationFile, 'f');
        if status
            copyStatus = 'COPIED';
            copied = true;
        else
            copyStatus = ['FAILED: ' message];
        end
    catch ME
        copyStatus = ['FAILED: ' ME.message];
    end

    records(end+1,:) = {sourceFile, destinationFile, category, copyStatus, fileInfo.bytes}; %#ok<AGROW>
end

function categoryFolder = classifyResultFile(fileName)
    name = lower(fileName);

    % Comparisons and per-class metric summaries are grouped together first.
    if contains(name, 'comparison') || contains(name, 'per_class_metrics') || ...
            contains(name, 'model_comparison')
        categoryFolder = '07_RESULTS_AND_MODEL_COMPARISONS';
    elseif contains(name, 'original_yamnet')
        categoryFolder = '04_ORIGINAL_YAMNET_BASELINE';
    elseif contains(name, 'frozen_yamnet')
        categoryFolder = '05_FROZEN_YAMNET_SVM';
    elseif contains(name, 'finetun') || contains(name, 'checkpoint') || ...
            contains(name, 'learnable_parameter') || contains(name, 'training_setup') || ...
            contains(name, 'step6') || contains(name, 'yamnet_architecture')
        categoryFolder = '06_YAMNET_FINE_TUNING';
    elseif contains(name, 'duration') || contains(name, 'manifest') || ...
            contains(name, 'split')
        categoryFolder = '02_DATASET_AND_PREPROCESSING';
    elseif contains(name, 'feature') || contains(name, 'spectrogram') || ...
            contains(name, 'mfcc') || contains(name, 'logmel')
        categoryFolder = '03_SIGNAL_PROCESSING_AND_FEATURES';
    else
        categoryFolder = '07_RESULTS_AND_MODEL_COMPARISONS';
    end
end

function writeTextFile(filePath, lines)
    parentFolder = fileparts(filePath);
    if ~isfolder(parentFolder)
        mkdir(parentFolder);
    end
    fileID = fopen(filePath, 'w');
    if fileID < 0
        warning('Could not create text file: %s', filePath);
        return;
    end
    cleanup = onCleanup(@() fclose(fileID)); %#ok<NASGU>
    for i = 1:numel(lines)
        fprintf(fileID, '%s\r\n', lines{i});
    end
end
