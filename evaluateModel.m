function metrics = evaluateModel(trueLabels, predictedLabels, classNames)

if numel(trueLabels) ~= numel(predictedLabels)
    error('trueLabels and predictedLabels must contain the same number of elements.');
end

noOfMatches = sum(trueLabels == predictedLabels);
totalNoOfLabels = numel(trueLabels);

accuracy = noOfMatches/totalNoOfLabels;

cm = confusionmat(trueLabels, predictedLabels, "Order", categorical(classNames));

recall = diag(cm) ./ sum(cm, 2);
precision = diag(cm) ./ sum(cm, 1)';
confusionchart(cm, classNames);

fprintf('Overall accuracy: %.1f%%\n', accuracy*100);
for k = 1:numel(classNames)
    fprintf('%-10s recall = %.2f, precision = %.2f\n', ...
        classNames(k), recall(k), precision(k));
end

metrics.accuracy = accuracy;
metrics.confusion = cm;
metrics.recall = recall;
metrics.precision = precision;
metrics.classNames = classNames;
