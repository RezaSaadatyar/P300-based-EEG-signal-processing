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
path = "D:\P300-based-EEG-signal-processing\Data\";
% [filenames, path] = uigetfile({'*.mat', 'mat file'; '*.*', 'All Files'}, 'File Selection', ...
%     'multiselect', 'on');
%% ------------------------- Step 2: Filtering all runs -------------------------------
fs = 240;  % Define sampling frequency
order = 10;  
f_low = 0.5;
f_high = 20;
notch_freq = 50;
notch_filter = 'off';
filter_active = 'on';
design_method = "FIR";      % IIR, FIR
type_filter = "bandpass";   % low, high, bandpass
%% ------------------------- Step 3: Downsampling parameters --------------------------
fd = 2 * f_high;
[p, q]= rat(fd / fs);
%% ---------------- Step 4: Detect target trials from non target trials ---------------
count1 = 0;
count2 = 0;
time_trial = 600; % Define the duration of each trial in milliseconds (e.g., 600 is ms)
duration_trial = round(time_trial/1000 * fs);
% select_channel = 1:64; 
select_channel = [9 11 13 34 49 51 53 56 60 62]; % fz,Cz,Pz,Oz,C3,C4,P3,P4,Po7,Po8

% for i = 1:length(filenames)
for i = 1:5
    load([char(path) 'AAS010R0' num2str(i)]); % Load the data from the selected mat file
    % load([path filenames{i}]); % Load the data from the selected mat file

    for j = 1:max(trialnr)
        % Get the start time of each trial
        ind = find(trialnr==j);
        % Start point of ith trial untill of ith trial
        data  = signal(ind(1):ind(1) + (duration_trial - 1), select_channel);
        % ------------------------------- Filtering -----------------------------------
        data = filtering(data, f_low, f_high, order, fs, notch_freq, filter_active, ...
            notch_filter, type_filter, design_method);
        % ------------------------------- Downsampling -------------------------------- 
        data = resample(data, p, q);       
        % --------------- Detect target trials from non target trials -----------------
        if max(StimulusType(ind)) == 1 % type of ith trial
            count1 = count1 + 1;
            target_data(:, count1) = data(:);% target trials
        elseif max(StimulusType(ind)) == 0 % type of ith trial
            count2 = count2 + 1;
            non_target_data(:, count2) = data(:); % non-target trials
        end
    end
end

% Balance dataset
ind = randperm(size(non_target_data, 2), size(target_data, 2));
non_target_data = non_target_data(:, ind);

% Combine target & non target data
data = [target_data, non_target_data];
labels = [ones(1, size(target_data, 2)), 2 * ones(1, size(non_target_data, 2))];
%% ----------------------------- Step 5: Model training -------------------------------
k_fold = 5;
c = [0.1 1 10 15 50 100];
sigma = [1 5 10 50 80 100 120];
result = svm_linear_optimal(data, labels, k_fold, c);
% result = svm_kernel_optimal(data, labels, k_fold, c, sigma);
