function plotStress(readings, alertIdx, threshold)
time = (1:numel(readings)) * 0.48;

figure("Color", "k");
theme(gcf, "dark");

plot(time, readings, "-b", "LineWidth", 1.5); hold on
yline(threshold, "--r");

plot(time(alertIdx), readings(alertIdx), "ro", "MarkerFaceColor", "r");

xlabel("Time(s)");
ylabel("Stress(%)");

title("Poultry Stress Readings");
xlim([0 max(time)]);
ylim([0, 100]);


hold off