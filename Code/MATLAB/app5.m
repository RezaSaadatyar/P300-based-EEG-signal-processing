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
    'multiselect', 'on');

fs = 240;  % Define sampling frequency
time_trial = 600; % Define the duration of each trial in milliseconds (e.g., 600 is ms)
duration_trial = round(time_trial/1000 * fs);
%% ------------------------- Step 2: Filtering all runs -------------------------------
f_low = 0.5;
f_high = 20;
order = 10;  
notch_freq = 50;
notch_filter = 'off';
filter_active = 'on';
design_method = "FIR";      % IIR, FIR
type_filter = "bandpass";   % low, high, bandpass
%% ------------------------- Step 3: Downsampling parameters --------------------------
fd = 2 * f_high;
[p, q]= rat(fd / fs);
%%
count1 = 0;
count2 = 0;
for i = 1:length(filenames) 
    load([path filenames{i}]); % Load the data from the selected mat file
    for j = 1:max(trialnr)
        % Get the start time of each trial
        ind = find(trialnr==j);
        % Start point of ith trial untill of ith trial
        data  = signal(ind(1):ind(1) + (duration_trial - 1), :);
        % Filtering
        filtered_data = filtering(data, f_low, f_high, order, fs, notch_freq, filter_active, ...
            notch_filter, type_filter, design_method);
        % Downsampling
        downsample_data = resample(filtered_data, p, q);       
        % Detect target trials from non target trials
        if max(StimulusType(ind)) == 1 % type of ith trial
            count1 = count1 + 1;
            data_target(:, count1) = downsample_data(:);% target trials
        elseif max(StimulusType(ind)) == 0 % type of ith trial
            count2 = count2 + 1;
            data_non_target(:, count2) = downsample_data(:); % non-target trials
        end
    end
end
%% Balance dataset
indx = randperm(size(data_non_target, 2), size(data_target, 2));
data_non_target= data_non_target(:, indx);

features = [data_target; data_non_target];
labels= [ones(1, size(data_target, 1)), 2*ones(1, size(data_non_target, 1))];
%% ------------------------------- Step 5: Classification -----------------------------
k_fold = 5;
num_neigh_knn = 3;
kernel_svm = 'linear';
distr_bayesian = 'normal';  % 'normal','kernel'
% 'linear','quadratic','diaglinear','diagquadratic','pseudolinear','pseudoquadratic'
discrimtype_lda = 'linear'; 
num_neurons_elm = 12;
Num_Neurons = 15;
num_center_rbf = 20;
sigma_pnn = 0.1;
type_pnn = 'Euclidean';      % 'Euclidean';'Correlation'
classifiation(features, labels, k_fold, num_neigh_knn, kernel_svm, distr_bayesian, ...
              discrimtype_lda, num_neurons_elm, num_center_rbf, sigma_pnn, type_pnn)
%% -------------------------------------- Plot ----------------------------------------
figure();
classes = unique(labels);
if size(features, 1) < size(features, 2); features = features'; end

for i = 1:numel(classes)
    if size(features, 1) == 2
        plot(features(labels == classes(i), 1), features(labels==classes(i), 2), 'o', ...
            'LineWidth', 1.5, 'MarkerSize', 4); hold on
        xlabel('Feature 1'); ylabel('Feature 2');
    elseif size(features, 1) > 2
        plot3(features(labels == classes(i), 1), features(labels == classes(i), 2), ...
            features(labels == classes(i), 3), 'o', 'LineWidth', 1.5, 'MarkerSize', 4);
        hold on
        xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3');
    end
end
grid on; til=legend("class1", "Class2"); 

figure
subplot(2, 1, 1)
plot(data(:, 11), 'linewidth', 2);
hold on
plot(filtered_data(:, 11), 'linewidth', 2);
legend('Raw data', 'Filtered_data')
subplot(2, 1, 2)
plot(downsample_data(:, 11), 'linewidth', 2);
legend('Dowsampling')
