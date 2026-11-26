
%% PHASE 4B: VERIFY EXTRACTED FEATURE FILES

projectFolder = pwd;
featureRoot = fullfile(projectFolder, 'Features');

specRoot = fullfile(featureRoot, 'Spectrograms');
mfccRoot = fullfile(featureRoot, 'MFCC');
logmelRoot = fullfile(featureRoot, 'LogMel');

% Confirm that all feature folders exist
assert(isfolder(specRoot), 'Spectrogram folder not found.');
assert(isfolder(mfccRoot), 'MFCC folder not found.');
assert(isfolder(logmelRoot), 'LogMel folder not found.');

% Count MAT files recursively
specFiles = dir(fullfile(specRoot, '**', '*.mat'));
mfccFiles = dir(fullfile(mfccRoot, '**', '*.mat'));
logmelFiles = dir(fullfile(logmelRoot, '**', '*.mat'));

fprintf('\n===== PHASE 4 FEATURE VERIFICATION =====\n');

fprintf('Spectrogram files: %d\n', numel(specFiles));
fprintf('MFCC files:        %d\n', numel(mfccFiles));
fprintf('Log-mel files:     %d\n', numel(logmelFiles));

% Verify expected file counts
if numel(specFiles) == 345 && ...
        numel(mfccFiles) == 345 && ...
        numel(logmelFiles) == 345

    fprintf('\nSUCCESS: All 345 recordings have all three feature families.\n');

else
    warning('Feature file counts do not match the expected 345.');
end

% Inspect the first saved feature file in each family
fprintf('\n===== SAMPLE FILE CONTENTS =====\n');

fprintf('\nSpectrogram sample:\n');
whos('-file', fullfile(specFiles(1).folder, specFiles(1).name));

fprintf('\nMFCC sample:\n');
whos('-file', fullfile(mfccFiles(1).folder, mfccFiles(1).name));

fprintf('\nLog-mel sample:\n');
whos('-file', fullfile(logmelFiles(1).folder, logmelFiles(1).name));
