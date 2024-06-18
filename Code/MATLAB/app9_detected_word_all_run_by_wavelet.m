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
f_high = 0.5;
notch_freq = 50;
notch_filter = 'off';
filter_active = 'on';
design_method = "FIR";      % IIR, FIR
type_filter = "high";   % low, high, bandpass
%% ------------------------- Step 3: Wavelet parameters --------------------------
wavelet_name = 'db10';
num_levels = 3;
%% ---------------- Step 4: Detect target trials from non target trials ---------------
count1 = 0;
count2 = 0;
time_trial = 600; % Define the duration of each trial in milliseconds (e.g., 600 is ms)
duration_trial = round(time_trial/1000 * fs);
% select_channel = 1:64; 
select_channel = [9 11 13 34 49 51 53 56 60 62]; % fz,Cz,Pz,Oz,C3,C4,P3,P4,Po7,Po8

% for i = 1:length(filenames)
for j = 1:5
    load([char(path) 'AAS010R0' num2str(j)]); % Load the data from the selected mat file
    % load([path filenames{i}]); % Load the data from the selected mat file

    for k = 1:max(trialnr)
        % Get the start time of each trial
        ind = find(trialnr==k);
        % Start point of ith trial untill of ith trial
        data  = signal(ind(1):ind(1) + (duration_trial - 1), select_channel);
        % ------------------------------ Filtering ------------------------------------
        data = filtering(data, f_low, f_high, order, fs, notch_freq, filter_active, ...
            notch_filter, type_filter, design_method);
        % -------------------------- Wavelet decompostion -----------------------------
        for ind_wav = 1:size(data, 2)
            [c, l] = wavedec(data(:, ind_wav), num_levels, wavelet_name);
            a(:, ind_wav) = c(1:l(1));
        end  
        % --------------- Detect target trials from non target trials -----------------
        if max(StimulusType(ind)) == 1 % type of ith trial
            count1 = count1 + 1;
            target_data(:, count1) = a(:);% target trials
        elseif max(StimulusType(ind)) == 0 % type of ith trial
            count2 = count2 + 1;
            non_target_data(:, count2) = a(:); % non-target trials
        end
    end
end
% ---------------------------------- Balance dataset ----------------------------------
ind = randperm(size(non_target_data, 2), size(target_data, 2));
non_target_data = non_target_data(:, ind);

% Combine target & non target data
data = [target_data, non_target_data];
labels = [ones(1, size(target_data, 2)), -1 * ones(1, size(non_target_data, 2))];
%% ----------------------------- Step 5: Model training -------------------------------
model = fitcsvm(data', labels, 'Standardize', 1, 'BoxConstraint', 0.1);
% model = fitcsvm(data', labels, 'Standardize', 1, 'KernelFunction', 'rbf', 'KernelScale',...
%     100, 'BoxConstraint', 120);
%% ------- Step 6: Word detection in all runs using the training model training -------
% Let the user select a mat file containing EEG data
% [filenames, path] = uigetfile({'*.mat', 'mat file'; '*.*', 'All Files'}, 'File Selection', ...
%     'multiselect', 'on');

time_on = 0.1;         %  Active time of each character (sec)
num_sequance = 5;     % number of seqeunce
num_all_characters = 12;
lookup_tabel = ['AGMSY5', 'BHNTZ6', 'CIOU17', 'DJPV28', 'EKQW39', 'FLRX4_'];
% true_word = ['CAT', 'DOG', 'FISH', 'WATER', 'BOWL']; % Session 10
% true_word = ['HAT', 'HAT', 'GLOVE', 'SHOES', 'FISH', 'RAT']; % Session 11
true_word = ['FOOD', 'MOOT', 'HAM', 'PIE', 'CAKE', 'TUNA', 'ZYGOT', '4567'];% Session 12
% ---------------- Step 6.1: Detect number of characters in each run ------------------
detected_word = [];
% for i = 1:length(filenames)
for i = 1:8
    load([char(path) 'AAS012R0' num2str(i)]); % Load the data from the selected mat file
    % load([path filenames{i}]); % Load the data from the selected mat file

    indx = find(PhaseInSequence==2);
    id = find(PhaseInSequence((indx - 1))==1); % Detect number of characters
    strartpoints = indx(id);                   % Detect start point each of character
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
        for k = 1: num_all_characters * num_sequance
            ind_trial= find(trialnr==trials(k));
            % Start point of ith trial untill of ith trial
            data = signal(ind_trial(1):ind_trial(1) + duration_trial - 1, select_channel);
            % ------------------------------- Filtering -------------------------------
            data = filtering(data, f_low, f_high, order, fs, notch_freq, filter_active,...
                notch_filter, type_filter, design_method);
            % ------------------------- Wavelet decompostion --------------------------
            for ind_wav = 1:size(data, 2)
                [c, l] = wavedec(data(:, ind_wav), num_levels, wavelet_name);
                a(:, ind_wav) = c(1:l(1));
            end
            % ------------------------------ Test model -------------------------------
            [~, distacne]= predict(model, a(:)');
            ind_stim = max(StimulusCode(ind_trial(1):ind_trial(1) + time_on * fs - 1));
            score(ind_stim) = score(ind_stim) + distacne(2);
        end
        % ---------------------------- target row and column --------------------------
        [~, col] = max(score(1:6));
        [~, row] = max(score(7:12));

        detect(j) = lookup_tabel(sub2ind([6 6], row, col));   % target character  
    end
    detected_word = [detected_word, detect];
    fprintf('Detected word by SVM: %s\n', detect);
    detect = [];
end
accuracy = sum(detected_word==true_word) / numel(true_word) *100;
disp(['Accuracy: ',num2str(accuracy)])