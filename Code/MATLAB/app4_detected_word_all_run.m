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

fs = 240;         % Define sampling frequency
time_trial = 600; % Define the duration of each trial in milliseconds (e.g., 600 is ms)
time_on = 0.1;    %  Active time of each character (sec)
time_peak = 320;  % Millisecend
num_sequance = 15;% number of seqeunce
select_channel = 11; % Cz
num_all_characters = 12;
duration_trial = round(time_trial/1000 * fs);
lookup_tabel = ['AGMSY5', 'BHNTZ6', 'CIOU17', 'DJPV28', 'EKQW39', 'FLRX4_'];
true_word = ['CAT', 'DOG', 'FISH', 'WATER', 'BOWL']; % Session 10
% true_word = ['HAT', 'HAT', 'GLOVE', 'SHOES', 'FISH', 'RAT']; % Session 11
% true_word = ['FOOD', 'MOOT', 'HAM', 'PIE', 'CAKE', 'TUNA', 'ZYGOT', '4567'];% Session 12
%% Detect number of characters in each run
detected_word = [];
for i = 1:length(filenames)
    load([path filenames{i}]); % Load the data from the selected mat file
    indx = find(PhaseInSequence==2);
    id = find(PhaseInSequence((indx - 1))==1); % Detect number of characters
    strartpoints = indx(id);                   % Detect start point each of character
    num_characters = numel(strartpoints);      % Number of characters

    for j = 1:num_characters
        if j < num_characters     % Find the index of trials for each character
            ind = find(samplenr > strartpoints(j) & samplenr < strartpoints(j + 1));
        else
            ind = find(samplenr > strartpoints(j));
        end
        ind(Flashing(ind)==0) = [];
        trials = unique(trialnr(ind));

        % Split trials related each character
        score= zeros(1, num_all_characters);
        for k = 1: num_all_characters * num_sequance
            ind_trial= find(trialnr==trials(k));
            % Start point of ith trial untill of ith trial
            sig = signal(ind_trial(1):ind_trial(1) + duration_trial - 1, select_channel);
            ind_stim = max(StimulusCode(ind_trial(1):ind_trial(1) + time_on * fs - 1));
            score(ind_stim) = score(ind_stim) + sig(round(time_peak / 1000 * fs));
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
