% ================================ P300 (2024-2025) ===================================
% ========================= Presented by: Reza Saadatyar ==============================
% ======================== E-mail: Reza.Saadatyar@outlook.com =========================
clc;            % Clear command window
clear;          % Clear workspace variables
close all;      % Close all figures
%% -------------------------------- Step 1: Load Data ---------------------------------
% Add the current directory and its subfolders to the MATLAB search path
addpath(genpath(cd)) 
% Let the user select a mat file containing EEG data
[filenames, path] = uigetfile({'*.mat', 'mat file'; '*.*', 'All Files'}, 'File Selection', ...
    'multiselect', 'off');

fs = 240;  % Define sampling frequency
%% ----------------Step 2: Detect target trials from non target trials ----------------
count1 = 0;
count2 = 0;
time_trial = 600; % Define the duration of each trial in milliseconds (e.g., 600 is ms)
duration_trial = round(time_trial/1000 * fs);

for i = 1:length(filenames) 
    load([path filenames{i}]); % Load the data from the selected mat file
    for j = 1:max(trialnr)
        % Get the start time of each trial
        ind = find(trialnr==j);
        % Start point of ith trial untill of ith trial
        data  = signal(ind(1):ind(1) + (duration_trial - 1), :);

        % Detect target trials from non target trials
        if max(StimulusType(ind)) == 1 % type of ith trial
            count1 = count1 + 1;
            data_target(:, :, count1) = data;% target trials
        elseif max(StimulusType(ind)) == 0 % type of ith trial
            count2 = count2 + 1;
            data_non_target(:, :, count2) = data; % non-target trials
        end
    end
end
%% ------------------------------- Step 3: Plot Result --------------------------------
target_a_trial = data_target(:, 1, 1);
non_target_a_trial = data_non_target(:, 1, 1);
target_mean = mean(data_target(:, 1, :), 3);
non_target_mean = mean(data_non_target(:, 1,:),3);
time = linspace(0, time_trial / 1000, size(data_target, 1));

figure
subplot(211)
plot(time, target_mean, 'b', 'linewidth', 2)
hold on
plot(time, non_target_mean, 'r', 'linewidth', 2)
legend('Target', 'non-Target')

subplot(212)
plot(time, target_a_trial, 'b', 'linewidth', 2)
hold on
plot(time, non_target_a_trial, 'r', 'linewidth', 2)
legend('Target', 'non-Target')
xlabel("Time (Sec)")