function [vector] = stressFromScores(scores, classNames)

stressedIndex = find(strcmp(classNames, "stressed"), 1);
if isempty(stressedIndex)
    error('classNames must contain "stressed".');
end

vector = scores(:, stressedIndex) * 100;

end