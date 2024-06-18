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
f_high = 30;
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

    for k = 1:max(trialnr)
        % Get the start time of each trial
        ind = find(trialnr==k);
        % Start point of ith trial untill of ith trial
        sig  = signal(ind(1):ind(1) + (duration_trial - 1), select_channel);
        % ------------------------------ Filtering ------------------------------------
        sig = filtering(sig, f_low, f_high, order, fs, notch_freq, filter_active, ...
            notch_filter, type_filter, design_method);
        % ------------------------------ Downsampling ---------------------------------
        sig = resample(sig, p, q);
        % --------------- Detect target trials from non target trials -----------------
        if max(StimulusType(ind)) == 1 % type of ith trial
            count1 = count1 + 1;
            target_data(:, count1) = sig(:);% target trials
        elseif max(StimulusType(ind)) == 0 % type of ith trial
            count2 = count2 + 1;
            non_target_data(:, count2) = sig(:); % non-target trials
        end
    end
end
%% ----------------------------- Step 5: Model training -------------------------------
type_classifer = "svm";    % svm, lda, mlp
ind= 1:size(target_data, 2):size(non_target_data, 2);
mdl = {};
for j = 1:length(ind)
    data2_sub = non_target_data(:, ind(j):ind(j) + size(target_data, 2) - 1);
    sig = [target_data, data2_sub];  % Combine target & non target data
    labels = [ones(1, size(target_data, 2)), -1 * ones(1, size(data2_sub, 2))];

    if strcmpi(type_classifer, 'SVM')
        mdl{j} = fitcsvm(sig', labels, 'Standardize', 1);
        % model{j} = fitcsvm(data', labels, 'Standardize', 1, 'KernelFunction', 'rbf', ...
        %     'KernelScale', 120, 'BoxConstraint', 100);
    elseif strcmpi(type_classifer, 'LDA')
        mdl{j} = fitcdiscr(sig', labels);
    elseif strcmpi(type_classifer, 'MLP')
        hiddenLayerSize = 10;     % Size of the hidden layer
        net = feedforwardnet(hiddenLayerSize);
        % Configure the training process
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
        % % Train Network
        [mdl{j}, ~] = train(net, sig, labels);
    end
end
%% ------- Step 6: Word detection in all runs using the training model training -------
time_on = 0.1;        %  Active time of each character (sec)
num_sequance = 5;     % number of seqeunce
detected_word = [];
num_all_characters = 12;
lookup_tabel = ['AGMSY5', 'BHNTZ6', 'CIOU17', 'DJPV28', 'EKQW39', 'FLRX4_'];
true_word = ['FOOD', 'MOOT', 'HAM', 'PIE', 'CAKE', 'TUNA', 'ZYGOT', '4567'];% Session 12

% Let the user select a mat file containing EEG data
% [filenames, path] = uigetfile({'*.mat', 'mat file'; '*.*', 'All Files'}, 'File Selection', ...
%     'multiselect', 'on');
% ---------------- Step 6.1: Detect number of characters in each run ------------------
% for i = 1:length(filenames)
for i = 1:8
    load([char(path) 'AAS012R0' num2str(i)]); % Load the data from the selected mat file
    % load([path filenames{i}]); % Load the data from the selected mat file
    ind = find(PhaseInSequence == 2);
    id_ = find(PhaseInSequence((ind - 1)) == 1); % Detect number of characters
    strartpoints = ind(id_);                   % Detect start point each of character
    num_characters = numel(strartpoints);      % Number of characters
    % ------------- Step 6.2: Find the index of trials for each character -------------
    for j = 1:num_characters
        if j < num_characters     % Find the index of trials for each character
            ind = find(samplenr > strartpoints(j) & samplenr < strartpoints(j + 1));
        else
            ind = find(samplenr > strartpoints(j));
        end
        ind(Flashing(ind)==0) = [];
        trials = unique(trialnr(ind));
        % --------------- Step 6.3: Split trials related each character ---------------
        score= zeros(1, num_all_characters);
        for k = 1:num_all_characters * num_sequance
            ind_trial= find(trialnr==trials(k));
            % Start point of ith trial untill of ith trial
            sig = signal(ind_trial(1):ind_trial(1) + duration_trial - 1, select_channel);
            % ----------------------------- Filtering ---------------------------------
            sig = filtering(sig, f_low, f_high, order, fs, notch_freq, filter_active, ...
                notch_filter, type_filter, design_method);
            % ----------------------------- Downsampling ------------------------------
            sig = resample(sig, p, q);
            if strcmpi(type_classifer, 'MLP')
                for r = 1:size(mdl, 2)
                 dist(r) = mdl{r}(sig(:));
                end 
            else
                for r = 1:size(mdl, 2)
                    [~, distacne] = predict(mdl{r}, sig(:)');
                    dist(r) = distacne(2);
                end
            end

            ind_stim = max(StimulusCode(ind_trial(1):ind_trial(1) + time_on * fs - 1));
            score(ind_stim) = score(ind_stim) + sum(dist);
        end
        % ----------------------------- target row and column -------------------------
        [~, col] = max(score(1:6));
        [~, row] = max(score(7:12));
        detect(j) = lookup_tabel(sub2ind([6 6], row, col));   % target character
    end
    detected_word = [detected_word, detect];
    fprintf('Detected word by %s: %s\n', type_classifer, detect);
    detect=[];
end
% -------------------------------------------------------------------------------------
accuracy = sum(detected_word==true_word) / numel(true_word) *100;
disp(['Accuracy: ', num2str(accuracy)])