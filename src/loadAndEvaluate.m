
function results = loadAndEvaluate(csvFile, classNames)

% Step 0: Define the exact CSV column names
trueCol = 'TrueClass';
predCol = 'PredictedClass';
splitCol = 'DatasetSplit';

T = readtable(csvFile, ...
    'Delimiter', ',', ...
    'VariableNamingRule', 'preserve');


% Convert labels to categorical
trueLabels = categorical(string(T.(trueCol)), classNames);
predictedLabels = categorical(string(T.(predCol)), classNames);

% Check for missing labels
if any(ismissing(trueLabels)) || any(ismissing(predictedLabels))

    disp(unique(string(T.(trueCol))));
    disp(unique(string(T.(predCol))));

    error('Label mismatch: Check class spelling.');
end

% Evaluate model
results = evaluateModel(trueLabels, predictedLabels, classNames);

end
