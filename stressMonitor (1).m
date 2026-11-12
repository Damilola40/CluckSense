function [alerts, alertIdx] = stressMonitor(readings, threshold, streakLimit)

% STRESSMONITOR Raise alerts when stress stays high for a sustained period.
%
%   alerts = stressMonitor(readings)
%   alerts = stressMonitor(readings, threshold)
%   alerts = stressMonitor(readings, threshold, streakLimit)
%   [alerts, alertIdx] = stressMonitor(___)
%
%   readings    - vector of stress percentages (0-100), one every 10 s
%   threshold   - readings above this count as "high" (default 70)
%   streakLimit - consecutive high readings needed to alert (default 30)
%
%   alerts      - cell array of text alerts, one per alert raised
%   alertIdx    - reading numbers at which alerts were raised
%
%   One alert is raised per stress episode. After an alert, no new alert
%   is raised until a reading drops to the threshold or below.

arguments
    readings (1,:) {mustBeNumeric, mustBeVector}
    threshold (1,1) {mustBeNumeric, mustBeBetween(threshold, 0, 100)} = 70
    streakLimit (1,1) {mustBeInteger, mustBePositive} = 30
end

alerts = {};
alertIdx = [];
consecutiveHigh = 0;
alreadyAlerted = false;

for i = 1:numel(readings)
    if readings(i) > threshold
        consecutiveHigh = consecutiveHigh + 1;
    else
        consecutiveHigh = 0;
        alreadyAlerted = false;   % stress has cleared, allow a new alert
    end

    if consecutiveHigh >= streakLimit && ~alreadyAlerted
        elapsedS = i * 10;
        alerts{end+1} = sprintf( ...
            'ALERT at reading #%d (%d s): %d consecutive readings above %g%%', ...
            i, elapsedS, streakLimit, threshold); %#ok<AGROW>
        alertIdx(end+1) = i; %#ok<AGROW>
        alreadyAlerted = true;
    end
end
end