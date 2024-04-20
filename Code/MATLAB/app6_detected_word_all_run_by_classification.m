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
%% ---------------- Step 4: Detect target trials from non target trials ---------------
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
        % Filtering
        data = filtering(data, f_low, f_high, order, fs, notch_freq, filter_active, ...
            notch_filter, type_filter, design_method);
        % Downsampling
        data = resample(data, p, q);       
        % Detect target trials from non target trials
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
labels = [ones(1, size(target_data, 2)), -1 * ones(1, size(non_target_data, 2))];
%% ----------------------------- Step 5: Model training -------------------------------
model = fitcsvm(data', labels, 'Standardize', 1);
%% ------- Step 6: Word detection in all runs using the training model training -------
% Let the user select a mat file containing EEG data
[filenames, path] = uigetfile({'*.mat', 'mat file'; '*.*', 'All Files'}, 'File Selection', ...
    'multiselect', 'on');

time_on = 0.1;         %  Active time of each character (sec)
num_sequance = 15;     % number of seqeunce
select_channel = 1:64; % Cz
num_all_characters = 12;
lookup_tabel = ['AGMSY5', 'BHNTZ6', 'CIOU17', 'DJPV28', 'EKQW39', 'FLRX4_'];
% true_word = ['CAT', 'DOG', 'FISH', 'WATER', 'BOWL']; % Session 10
% true_word = ['HAT', 'HAT', 'GLOVE', 'SHOES', 'FISH', 'RAT']; % Session 11
true_word = ['FOOD', 'MOOT', 'HAM', 'PIE', 'CAKE', 'TUNA', 'ZYGOT', '4567'];% Session 12
% ---------------- Step 6.1: Detect number of characters in each run ------------------
detected_word = [];
for i = 1:length(filenames)
    load([path filenames{i}]); % Load the data from the selected mat file
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
            sig = signal(ind_trial(1):ind_trial(1) + duration_trial - 1, select_channel);
            % Filtering
            sig = filtering(sig, f_low, f_high, order, fs, notch_freq, filter_active,...
                notch_filter, type_filter, design_method);
            % Downsampling
            sig = resample(sig, p, q);  
            sig = sig(:);
            [~, distacne]= predict(model, sig');
            ind_stim = max(StimulusCode(ind_trial(1):ind_trial(1) + time_on * fs - 1));
            score(ind_stim) = score(ind_stim) + distacne(2);
        end
        % target row and column
        [~, col] = max(score(1:6));
        [~, row] = max(score(7:12));

        detect(j) = lookup_tabel(sub2ind([6 6], row, col));   % target character  
    end
    detected_word = [detected_word, detect];
    disp(['Detected word: ', detect])
    detect = [];
end
accuracy = sum(detected_word==true_word) / numel(true_word) *100;
disp(['Accuracy: ',num2str(accuracy)])
%% ------------------------------- Step 5: Classification -----------------------------
% k_fold = 5;
% num_neigh_knn = 3;
% kernel_svm = 'linear';
% distr_bayesian = 'normal';  % 'normal','kernel'
% % 'linear','quadratic','diaglinear','diagquadratic','pseudolinear','pseudoquadratic'
% discrimtype_lda = 'linear'; 
% num_neurons_elm = 12;
% Num_Neurons = 15;
% num_center_rbf = 20;
% sigma_pnn = 0.1;
% type_pnn = 'Euclidean';      % 'Euclidean';'Correlation'
% classifiation(data, labels, k_fold, num_neigh_knn, kernel_svm, distr_bayesian, ...
%               discrimtype_lda, num_neurons_elm, num_center_rbf, sigma_pnn, type_pnn)
% %% -------------------------------------- Plot ----------------------------------------
% figure();
% classes = unique(labels);
% if size(features, 1) < size(features, 2); features = features'; end
% 
% for i = 1:numel(classes)
%     if size(features, 1) == 2
%         plot(features(labels == classes(i), 1), features(labels==classes(i), 2), 'o', ...
%             'LineWidth', 1.5, 'MarkerSize', 4); hold on
%         xlabel('Feature 1'); ylabel('Feature 2');
%     elseif size(features, 1) > 2
%         plot3(features(labels == classes(i), 1), features(labels == classes(i), 2), ...
%             features(labels == classes(i), 3), 'o', 'LineWidth', 1.5, 'MarkerSize', 4);
%         hold on
%         xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3');
%     end
% end
% grid on; til=legend("class1", "Class2"); 
% 
% figure
% subplot(2, 1, 1)
% plot(data(:, 11), 'linewidth', 2);
% hold on
% plot(filtered_data(:, 11), 'linewidth', 2);
% legend('Raw data', 'Filtered_data')
% subplot(2, 1, 2)
% plot(downsample_data(:, 11), 'linewidth', 2);
% legend('Dowsampling')