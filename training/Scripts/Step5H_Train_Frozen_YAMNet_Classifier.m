
%% STEP 5H: TRAIN FROZEN YAMNET THREE-CLASS BASELINE
% CluckSense Smart Poultry
% YAMNet remains frozen.
% A separate linear SVM learns the three poultry classes.

clc;
clear;
close all;

%% 1. LOAD FROZEN YAMNET EMBEDDINGS

projectFolder = pwd;

embeddingFile = fullfile(projectFolder, ...
    "Features", "YAMNet_Embeddings", ...
    "Frozen_YAMNet_Embeddings.mat");

resultsFolder = fullfile(projectFolder, "Results");

if ~isfile(embeddingFile)
    error("Frozen YAMNet embedding file not found.");
end

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

load(embeddingFile, ...
    "embeddings", "embeddingTable", ...
    "embeddingDimension", "embeddingLayer");

fprintf("Frozen YAMNet embeddings loaded.\n");
fprintf("Embedding dimensions: %d x %d\n", ...
    size(embeddings,1), size(embeddings,2));

%% 2. VERIFY DATA

if size(embeddings,1) ~= height(embeddingTable)
    error("Embedding rows do not match metadata records.");
end

if size(embeddings,2) ~= embeddingDimension
    error("Unexpected embedding dimension.");
end

if any(~isfinite(embeddings(:)))
    error("Embeddings contain NaN or Inf values.");
end

if any(embeddingTable.ProcessingStatus ~= "Success")
    error("Some recordings do not have successful embeddings.");
end

%% 3. DEFINE CLASS ORDER

classOrder = categorical([ ...
    "Background_Noise"
    "Healthy"
    "Respiratory_Abnormal"]);

labels = categorical(embeddingTable.ClassLabel, ...
    string(classOrder));

splitNames = embeddingTable.DatasetSplit;

trainIndices = splitNames == "Training";
validationIndices = splitNames == "Validation";
testIndices = splitNames == "Testing";

XTrain = double(embeddings(trainIndices,:));
YTrain = labels(trainIndices);

XValidation = double(embeddings(validationIndices,:));
YValidation = labels(validationIndices);

XTest = double(embeddings(testIndices,:));
YTest = labels(testIndices);

fprintf("\nDATASET SPLIT SUMMARY\n");
fprintf("Training: %d\n",size(XTrain,1));
fprintf("Validation: %d\n",size(XValidation,1));
fprintf("Testing: %d\n",size(XTest,1));

%% 4. STANDARDIZE FEATURES USING TRAINING DATA ONLY

% The validation and test data must not influence
% the training mean or standard deviation.

featureMean = mean(XTrain,1);
featureStd = std(XTrain,0,1);

% Avoid division by zero for constant features.
featureStd(featureStd == 0) = 1;

XTrainStandardized = ...
    (XTrain - featureMean) ./ featureStd;

XValidationStandardized = ...
    (XValidation - featureMean) ./ featureStd;

XTestStandardized = ...
    (XTest - featureMean) ./ featureStd;

%% 5. CALCULATE CLASS-BALANCED ERROR COSTS

numberOfClasses = numel(classOrder);

trainingClassCounts = zeros(numberOfClasses,1);

for k = 1:numberOfClasses
    trainingClassCounts(k) = sum(YTrain == classOrder(k));
end

if any(trainingClassCounts == 0)
    error("At least one class is missing from training data.");
end

% Penalize errors on less frequent classes more strongly.
costMatrix = zeros(numberOfClasses);

for i = 1:numberOfClasses
    for j = 1:numberOfClasses
        if i ~= j
            costMatrix(i,j) = 1 / trainingClassCounts(i);
        end
    end
end

fprintf("\nTRAINING CLASS COUNTS\n");

for k = 1:numberOfClasses
    fprintf("%s: %d\n", ...
        string(classOrder(k)), trainingClassCounts(k));
end

%% 6. TRAIN LINEAR SVM ECOC CLASSIFIER

fprintf("\nTraining frozen YAMNet baseline classifier...\n");

svmLearner = templateSVM( ...
    "KernelFunction","linear", ...
    "Standardize",false, ...
    "BoxConstraint",1);

baselineModel = fitcecoc( ...
    XTrainStandardized, ...
    YTrain, ...
    "Learners",svmLearner, ...
    "Coding","onevsone", ...
    "ClassNames",classOrder, ...
    "Cost",costMatrix);

fprintf("Classifier training completed.\n");

%% 7. VALIDATION EVALUATION

validationPredictions = predict( ...
    baselineModel,XValidationStandardized);

validationAccuracy = mean( ...
    validationPredictions == YValidation) * 100;

fprintf("\nVALIDATION ACCURACY: %.2f%%\n", ...
    validationAccuracy);

%% 8. FINAL TEST EVALUATION

testPredictions = predict( ...
    baselineModel,XTestStandardized);

testAccuracy = mean( ...
    testPredictions == YTest) * 100;

fprintf("\n========================================\n");
fprintf("FROZEN YAMNET BASELINE TEST RESULTS\n");
fprintf("========================================\n");

fprintf("Test recordings: %d\n",numel(YTest));
fprintf("Correct predictions: %d\n", ...
    sum(testPredictions == YTest));
fprintf("Incorrect predictions: %d\n", ...
    sum(testPredictions ~= YTest));

fprintf("Test accuracy: %.2f%%\n",testAccuracy);

%% 9. CONFUSION MATRIX

confusionMatrix = confusionmat( ...
    YTest,testPredictions, ...
    "Order",classOrder);

disp("Confusion Matrix:");
disp(array2table(confusionMatrix, ...
    "VariableNames",cellstr(string(classOrder)), ...
    "RowNames",cellstr(string(classOrder))));


%% 10. PRECISION, RECALL AND F1-SCORE

truePositive = diag(confusionMatrix);
falsePositive = sum(confusionMatrix,1)' - truePositive;
falseNegative = sum(confusionMatrix,2) - truePositive;

% Ensure all metric variables are column vectors.
truePositive = truePositive(:);
falsePositive = falsePositive(:);
falseNegative = falseNegative(:);
classNamesColumn = string(classOrder(:));

precision = truePositive ./ ...
    max(truePositive + falsePositive,1);

recall = truePositive ./ ...
    max(truePositive + falseNegative,1);

f1Score = 2 .* precision .* recall ./ ...
    max(precision + recall,eps);

% Build the metrics table.
classMetrics = table( ...
    classNamesColumn, ...
    truePositive, ...
    falsePositive, ...
    falseNegative, ...
    precision * 100, ...
    recall * 100, ...
    f1Score * 100, ...
    'VariableNames',{ ...
    'Class', ...
    'TruePositive', ...
    'FalsePositive', ...
    'FalseNegative', ...
    'PrecisionPercent', ...
    'RecallPercent', ...
    'F1ScorePercent'});

disp("Per-class performance:");
disp(classMetrics);

macroF1 = mean(f1Score) * 100;

fprintf("Macro F1-score: %.2f%%\n",macroF1);


%% 11. SAVE PREDICTIONS

testMetadata = embeddingTable(testIndices,:);

% Force all variables to be column vectors.
predictionFileNames = string(testMetadata.FileName(:));
actualClasses = YTest(:);
predictedClasses = testPredictions(:);

% Verify that all prediction variables have equal lengths.
if numel(predictionFileNames) ~= numel(actualClasses) || ...
        numel(actualClasses) ~= numel(predictedClasses)

    error("Prediction variables have inconsistent dimensions.");

end

predictionTable = table( ...
    predictionFileNames, ...
    actualClasses, ...
    predictedClasses, ...
    'VariableNames',{ ...
    'FileName', ...
    'ActualClass', ...
    'PredictedClass'});

predictionFile = fullfile(resultsFolder, ...
    "Frozen_YAMNet_Test_Predictions.csv");

writetable(predictionTable,predictionFile);

fprintf("\nTest predictions saved successfully.\n");

%% 12. SAVE MODEL AND RESULTS

modelFile = fullfile(resultsFolder, ...
    "Frozen_YAMNet_Baseline_Model.mat");

save(modelFile, ...
    "baselineModel", ...
    "featureMean", ...
    "featureStd", ...
    "embeddingDimension", ...
    "embeddingLayer", ...
    "classOrder", ...
    "validationAccuracy", ...
    "testAccuracy", ...
    "confusionMatrix", ...
    "classMetrics", ...
    "macroF1", ...
    "-v7.3");

metricsFile = fullfile(resultsFolder, ...
    "Frozen_YAMNet_Baseline_Class_Metrics.csv");

writetable(classMetrics,metricsFile);

%% 13. DISPLAY FINAL SUMMARY

fprintf("\n========================================\n");
fprintf("STEP 5H COMPLETED\n");
fprintf("========================================\n");

fprintf("Validation accuracy: %.2f%%\n", ...
    validationAccuracy);

fprintf("Test accuracy: %.2f%%\n", ...
    testAccuracy);

fprintf("Macro F1-score: %.2f%%\n",macroF1);

fprintf("\nSaved model:\n%s\n",modelFile);

fprintf("\nSaved predictions:\n%s\n",predictionFile);

fprintf("\nSaved class metrics:\n%s\n",metricsFile);
