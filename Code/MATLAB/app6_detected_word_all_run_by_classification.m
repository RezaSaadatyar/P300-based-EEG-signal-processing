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
[p, q] = rat(fd / fs);
%% ---------------- Step 4: Detect target trials from non target trials ---------------
count1 = 0;
count2 = 0;
time_trial = 600; % Define the duration of each trial in milliseconds (e.g., 600 is ms)
duration_trial = round(time_trial/1000 * fs);
% select_channel = 1:64;
select_channel = [9 11 13 34 49 51 53 56 60 62]; % fz, Cz, Pz, Oz, C3, C4, P3, P4, Po7, Po8

for i = 1:length(filenames)
    load([path filenames{i}]); % Load the data from the selected mat file
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
%% ----------------------------- Step 5: Model training -------------------------------
% Combine target & non target data
data = [target_data, non_target_data];
labels = [ones(1, size(target_data, 2)), -1 * ones(1, size(non_target_data, 2))];

type_classifer = "cnn";
if strcmpi(type_classifer, 'SVM')
    model = fitcsvm(data', labels, 'Standardize', 1);
    % model = fitcsvm(data', labels, 'Standardize', 1, 'KernelFunction', 'rbf', 'KernelScale',...
    %     100, 'BoxConstraint', 120);
elseif strcmpi(type_classifer, 'LDA')
    model = fitcdiscr(data', labels);
elseif strcmpi(type_classifer, 'KNN')
    model = fitcknn(data', labels, 'NumNeighbors', 18, 'Distance', 'euclidean', ...
        'DistanceWeight', 'inverse', 'NSMethod','exhaustive',...
        'StandardizeData', 0);
elseif strcmpi(type_classifer, 'MLP')
    hiddenLayerSize = 10;     % Size of the hidden layer
    net = feedforwardnet(hiddenLayerSize);
    % Configure the training process
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    % % Train Network
    [model, ~] = train(net, data, labels);
else
    numFeatures = size(data, 2); % Number of features per sample
    numSamples = size(data, 1); % Number of samples per class
    classes = 2; % Number of classes
    % data = [target_data; non_target_data];
    % data = reshape(data, numFeatures, 1, 1, numSamples * classes);
   data = reshape(data, numFeatures, 1, 1, numSamples);
    % labels = categorical([ones(1, numSamples), 2*ones(1, numSamples)]);
    labels = categorical([ones(1, numSamples), 2*ones(1, numSamples)]);
    % Split data into training and validation sets
    % idx = randperm(numSamples * classes);
    idx = randperm(numFeatures);
    trainData = data(:, :, :, idx(1:round(0.8*numSamples*classes)));
    trainLabels = labels(idx(1:round(0.8*numSamples*classes)));
    valData = data(:, :, :, idx(round(0.8*numSamples*classes)+1:end));
    valLabels = labels(idx(round(0.8*numSamples*classes)+1:end));

    % Define the 1D CNN architecture
    layers = [
        imageInputLayer([numFeatures, 1, 1])
        convolution2dLayer([3, 1], 8, 'Padding', 'same') % Treating as 1D convolution
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer([2, 1], 'Stride', [2, 1]) % Pooling along one dimension
        convolution2dLayer([3, 1], 16, 'Padding', 'same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer([2, 1], 'Stride', [2, 1])
        fullyConnectedLayer(classes)
        softmaxLayer
        classificationLayer
        ];

    % Training options
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.01, ...
        'MaxEpochs', 10, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {valData, valLabels}, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    % Train the network
    net = trainNetwork(trainData, trainLabels, layers, options);
end


% ------- Step 6: Word detection in all runs using the training model training -------
% Let the user select a mat file containing EEG data
% [filenames, path] = uigetfile({'*.mat', 'mat file'; '*.*', 'All Files'}, 'File Selection', ...
%     'multiselect', 'on');
path = 'D:\P300-based-EEG-signal-processing\Data\';
time_on = 0.1;         %  Active time of each character (sec)
num_sequance = 1;     % number of seqeunce
num_all_characters = 12;
lookup_tabel = ['AGMSY5', 'BHNTZ6', 'CIOU17', 'DJPV28', 'EKQW39', 'FLRX4_'];
% true_word = ['CAT', 'DOG', 'FISH', 'WATER', 'BOWL']; % Session 10
% true_word = ['HAT', 'HAT', 'GLOVE', 'SHOES', 'FISH', 'RAT']; % Session 11
true_word = ['FOOD', 'MOOT', 'HAM', 'PIE', 'CAKE', 'TUNA', 'ZYGOT', '4567'];% Session 12
% ---------------- Step 6.1: Detect number of characters in each run ------------------
detected_word = [];
% for i = 1:length(filenames)
for i = 1:8
    load([path 'AAS012R0' num2str(i)]); % Load the data from the selected mat file
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
            % ------------------------------- Filtering -------------------------------
            sig = filtering(sig, f_low, f_high, order, fs, notch_freq, filter_active,...
                notch_filter, type_filter, design_method);
            % ------------------------------ Downsampling -----------------------------
            sig = resample(sig, p, q);
            sig = sig(:);
            if strcmpi(type_classifer, 'MLP')
                distacne = model(sig);
            elseif strcmpi(type_classifer, 'CNN')
                a=reshape(sig, 1, 1, 1, numSamples);
                distacne = classify(net, a);
            else
                [~, distacne]= predict(model, sig');
            end
            ind_stim = max(StimulusCode(ind_trial(1):ind_trial(1) + time_on * fs - 1));
            score(ind_stim) = score(ind_stim) + distacne(end);
        end
        % --------------------------- target row and column ---------------------------
        [~, col] = max(score(1:6));
        [~, row] = max(score(7:12));

        detect(j) = lookup_tabel(sub2ind([6 6], row, col));   % target character
    end
    detected_word = [detected_word, detect];
    fprintf('Detected word by %s: %s\n', type_classifer, detect);
    detect = [];

end
accuracy = sum(detected_word==true_word) / numel(true_word) *100;
disp(['Accuracy: ',num2str(accuracy)])