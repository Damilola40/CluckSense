clc
clear

% TESTSTRESSMONITOR Quick checks for stressMonitor. Run this script.
% If nothing errors and the last line prints, all tests passed.

% 29 high readings: not enough for an alert
[~, idx] = stressMonitor(80*ones(1,29));
assert(isempty(idx), '29 high readings should give no alert');

% 30 high readings: one alert, at reading 30
[~, idx] = stressMonitor(80*ones(1,30));
assert(isequal(idx, 30), '30 high readings should alert at reading 30');

% 60 high readings: still ONE alert (one per episode)
[~, idx] = stressMonitor(80*ones(1,60));
assert(isequal(idx, 30), '60 high readings should give one alert at 30');

% 29 high, 1 low, 29 high: streak broken, no alert
[~, idx] = stressMonitor([80*ones(1,29) 10 80*ones(1,29)]);
assert(isempty(idx), 'A broken streak should give no alert');

% 30 high, 1 low, 30 high: two separate episodes, two alerts
[~, idx] = stressMonitor([80*ones(1,30) 10 80*ones(1,30)]);
assert(isequal(idx, [30 61]), 'Two episodes should give alerts at 30 and 61');

% Exactly 70 is not "above 70"
[~, idx] = stressMonitor(70*ones(1,100));
assert(isempty(idx), 'Readings of exactly 70 should not count as high');

% Custom threshold and streak limit
[~, idx] = stressMonitor(60*ones(1,5), 50, 5);
assert(isequal(idx, 5), 'Custom threshold and streak should work');

disp('All stressMonitor tests passed.');
