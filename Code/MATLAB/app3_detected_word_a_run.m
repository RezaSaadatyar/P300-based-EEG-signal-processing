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
%% ----------------------- Step 2: Detect the words in each run -----------------------
load([path filenames])
indx = find(PhaseInSequence==2);
% ---------------------- Step 2.1: Detect number of characters ------------------------
id = find(PhaseInSequence((indx - 1))==1); % Detect number of characters
strartpoints = indx(id);                   % Detect start point each of character
num_characters = numel(strartpoints);      % Number of characters
% -------------------------------------------------------------------------------------
fs = 240;  % Define sampling frequency
time_on = 0.1;       % Active time of each character (sec)
time_peak = 320;     % Millisecend
time_trial = 600; % Define the duration of each trial in milliseconds (e.g., 600 is ms)
num_sequance = 15;   % number of seqeunce
select_channel = 11; % Cz
num_all_characters = 12;
duration_trial = round(time_trial/1000 * fs);
lookup_tabel = ['AGMSY5', 'BHNTZ6', 'CIOU17', 'DJPV28', 'EKQW39', 'FLRX4_'];

for i = 1:num_characters      % First loop: Number of characters
    if i < num_characters     % Second loop: Find the index of trials for each character
        ind = find(samplenr > strartpoints(i) & samplenr < strartpoints(i + 1));
    else
        ind = find(samplenr > strartpoints(i));
    end
        ind(Flashing(ind)==0) = [];
        trials = unique(trialnr(ind));
        % ---------------- Step 2.2: Split trials related each character --------------
        score= zeros(1, num_all_characters);
        for j = 1: num_all_characters * num_sequance 
            ind_trial= find(trialnr==trials(j));
            % ------------- Start point of ith trial untill of ith trial --------------
            sig = signal(ind_trial(1):ind_trial(1) + duration_trial - 1, select_channel);
            ind_stim = max(StimulusCode(ind_trial(1):ind_trial(1) + time_on * fs - 1));
            score(ind_stim) = score(ind_stim) + sig(round(time_peak / 1000 * fs));
        end  
        % --------------------------- target row and column --------------------------- 
        [~, col] = max(score(1:6));
        [~, row] = max(score(7:12));

        detect(i) = lookup_tabel(sub2ind([6 6], row, col));   % target character      
end
disp(['Detected word: ', detect])
%% ------------------------------- Step 3: Plot Result --------------------------------
figure
plot(samplenr, StimulusCode, 'b', 'linewidth', 0.1)
hold on
plot(samplenr, PhaseInSequence, 'k', 'linewidth', 2)
plot(samplenr(strartpoints), PhaseInSequence(strartpoints), 'ro', 'linewidth', 2)