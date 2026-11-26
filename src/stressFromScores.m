function vector = stressFromScores(scores, classNames, targetClass)

stressedIndex = find(strcmp(string(classNames), string(targetClass)), 1);

if isempty(stressedIndex)
    error('targetClass must be in classNames');
end

if size(scores, 2) ~= numel(classNames)
    error('scores needs one column per class');
end

vector = scores(:, stressedIndex) * 100;

end
