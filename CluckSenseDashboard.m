function CluckSenseDashboard(stress, alertIdx, threshold, predictionsFile, classNames)
% CluckSenseDashboard  Simple MVP dashboard for CluckSense (Team PHAL).
%
% Usage (add at the end of runCluckSense.m):
%   CluckSenseDashboard(stress, alertIdx, threshold, predictionsFile, classNames);

secondsPerReading = 0.48;
stress = stress(:);
t = (1:numel(stress))' * secondsPerReading;
classNames = string(classNames(:))';          % 1 x 3 string row
nClass = numel(classNames);

% ---------- 1. Model metrics from the predictions CSV ----------
% Read the CSV as plain text lines (the Windows file paths contain
% backslashes, which readtable mistakes for column separators).
L = strip(readlines(predictionsFile));
L = L(L ~= "");
parts = split(L, ",");                         % n x 4: FileName,TrueClass,PredictedClass,DatasetSplit
parts = parts(parts(:, 2) ~= "TrueClass", :);  % drop the header row if there is one
parts = parts(strip(parts(:, 4)) == "Testing", :);
trueC = strip(parts(:, 2));
predC = strip(parts(:, 3));

cm = zeros(nClass);                            % rows = true, cols = predicted
for i = 1:nClass
    for j = 1:nClass
        cm(i, j) = sum(trueC == classNames(i) & predC == classNames(j));
    end
end
accuracy  = trace(cm) / sum(cm(:));
recall    = diag(cm) ./ sum(cm, 2);
precision = diag(cm) ./ sum(cm, 1)';

% ---------- 2. Alert summary ----------
nAlerts = numel(alertIdx);
if nAlerts > 0
    latestAlertTime = alertIdx(end) * secondsPerReading;
end

% ---------- 3. Layout ----------
fig = uifigure("Name", "CluckSense Dashboard", "Position", [100 100 1150 680]);
g = uigridlayout(fig, [3 3]);
g.RowHeight = {55, "1x", "1x"};
g.ColumnWidth = {"1x", "1x", 330};

titleLbl = uilabel(g, "Text", "CluckSense - Poultry Respiratory Stress Monitor (Team PHAL)", ...
    "FontSize", 22, "FontWeight", "bold");
titleLbl.Layout.Row = 1;
titleLbl.Layout.Column = [1 3];

% ---------- 4. Stress curve ----------
ax = uiaxes(g);
ax.Layout.Row = 2;
ax.Layout.Column = [1 2];
plot(ax, t, stress, "LineWidth", 1.2);
hold(ax, "on");
yline(ax, threshold, "--r", sprintf("Threshold = %g", threshold));
if nAlerts > 0
    plot(ax, t(alertIdx), stress(alertIdx), "rv", "MarkerFaceColor", "r", "MarkerSize", 8);
end
hold(ax, "off");
ylim(ax, [0 100]);
grid(ax, "on");
xlabel(ax, "Time (s)");
ylabel(ax, "Stress (%)");
title(ax, "Stress over time (simulated timeline from real recordings)");

% ---------- 5. Alert status panel ----------
p = uipanel(g, "Title", "Alert status");
p.Layout.Row = 2;
p.Layout.Column = 3;
pg = uigridlayout(p, [5 1]);
pg.RowHeight = {60, 35, 30, "1x", 25};

lamp = uilamp(pg);
lamp.Layout.Row = 1;
if nAlerts > 0
    lamp.Color = [0.85 0.1 0.1];
    statusText = "ALERT";
    detailText = sprintf("%d alert(s), latest at %.0f s", nAlerts, latestAlertTime);
else
    lamp.Color = [0.1 0.7 0.2];
    statusText = "NORMAL";
    detailText = "No alerts";
end

statusLbl = uilabel(pg, "Text", statusText, "FontSize", 20, ...
    "FontWeight", "bold", "HorizontalAlignment", "center");
statusLbl.Layout.Row = 2;

detailLbl = uilabel(pg, "Text", detailText, "HorizontalAlignment", "center");
detailLbl.Layout.Row = 3;

gauge = uigauge(pg, "semicircular");
gauge.Layout.Row = 4;
gauge.Limits = [0 100];
gauge.Value = min(max(stress(end), 0), 100);
gauge.ScaleColors = {[0.1 0.7 0.2], [0.9 0.2 0.2]};
gauge.ScaleColorLimits = [0 threshold; threshold 100];

gaugeLbl = uilabel(pg, "Text", "Latest stress reading", "HorizontalAlignment", "center");
gaugeLbl.Layout.Row = 5;

% ---------- 6. Metrics table ----------
mp = uipanel(g, "Title", sprintf("Model performance (test set) - overall accuracy %.1f%%", accuracy * 100));
mp.Layout.Row = 3;
mp.Layout.Column = [1 2];
mg = uigridlayout(mp, [1 1]);
mt = uitable(mg);
mt.Data = table(classNames', round(recall * 100, 1), round(precision * 100, 1), ...
    'VariableNames', {'Class', 'Recall (%)', 'Precision (%)'});

% ---------- 7. Confusion matrix ----------
cp = uipanel(g, "Title", "Confusion matrix (rows = true class)");
cp.Layout.Row = 3;
cp.Layout.Column = 3;
cg = uigridlayout(cp, [1 1]);
ct = uitable(cg);
ct.Data = cm;
ct.RowName = cellstr(classNames);
ct.ColumnName = cellstr(classNames);
end