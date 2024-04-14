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
[filename, path] = uigetfile({'*.mat', 'mat file'; '*.*', 'All Files'}, 'File Selection', ...
    'multiselect', 'off');
data = load([path filename]); % Load the data from the selected mat file
signal = data.signal; % Extract EEG data (number samples, number channel)
trialnr = data.trialnr;
flashing = data.Flashing;
stimulus_type = data.StimulusType;
stimulus_code = data.StimulusCode;
phase_sequence = data.PhaseInSequence;
ind = find(trialnr<=max(trialnr));
%% ------------------------------- Step 2: Result plot --------------------------------
% Plot original signal for num_channel 1
subplot(4, 1, 1);
plot(signal(:, 1));
title('Channel: 1');
legend('Raw signal', fontsize=8)

% Plot Flashing
subplot(4, 1, 2);
plot(flashing); hold on;
legend('Flashing', fontsize=8);

% Plot StimulusCode and PhaseInSequence
subplot(4, 1, 3);
plot(stimulus_code); hold on;
plot(max(stimulus_code) / 2 * stimulus_type(ind), 'r', 'linewidth', 2)
plot(phase_sequence, 'g', 'linewidth', 2);
legend('Stimulus Code', 'desired character', 'Phase in sequence', fontsize=8);

% Plot trialnr
subplot(4, 1, 4);
plot(trialnr, 'linewidth', 1);
ylabel('Trial Number', fontsize=10);
