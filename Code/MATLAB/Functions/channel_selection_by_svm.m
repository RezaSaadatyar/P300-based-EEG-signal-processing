function channel_selection_by_svm(data1, data2, k_fold, select_channel, removed_channels)
% ================================== (2023-2024) ======================================
% ======================== Presented by: Reza Saadatyar ===============================
% ====================== E-mail: Reza.Saadatyar@outlook.com ===========================
% Example:
% k_fold = 5;
% k_fold = 2;
% select_channel = 1:64;
% channel_selection_by_svm(target_data, non_target_data, k_fold, select_channel, removed_channels)
%% =============== Flowchart for the channel_selection_by_svm function ================
% Start
% 1. Initialize an empty array to store performance results.
% 2. Iterate over the range of removed channels:
%    a. Select the desired channels for analysis, removing the specified number of channels.
%    b. Iterate over each trial:
%       - Extract features from data1 and data2 based on the selected channels.
%       - Concatenate features from both datasets.
%       - Create label vectors for both datasets.
%    c. Perform k-fold cross-validation:
%       - Split the data into training and test sets.
%       - Train an SVM model using 'fitcsvm'.
%       - Predict labels for the test set.
%       - Calculate accuracy using confusion matrix.
%    d. Store the mean accuracy for the current number of removed channels.
%    e. Plot the performance curve.
% End
performance = [];
%% ====================================================================================
for num_channels = 1:length(removed_channels)
    select = select_channel;
    select(removed_channels(1:num_channels)) = [];
    
    for i=1:size(data2,3)
        x1 = data1(:, select, i);
        x2 = data2(:, select, i);
        
        features1(:, i) = x1(:);
        features2(:, i) = x2(:);
    end
    data = [features1, features2];
    labels = [ones(1, size(features1, 2)), 2 * ones(1, size(features2, 2))];
    indx = cvpartition(labels, 'k', k_fold);      % k-fold cross validation
    for k = 1:k_fold  % Perform k-fold cross-validation
        train_ind = indx.training(k); test_ind = indx.test(k);
        train_data = data(:, train_ind); train_labels = labels(train_ind);
        test_data = data(:, test_ind); test_labels = labels(test_ind);
        % ------------------------------------ SVM ------------------------------------
        model = fitcsvm(train_data', train_labels, 'Standardize', 1,'BoxConstraint', 0.01,...
                        'Verbose', 0);
        predict_labels = predict(model, test_data');
        confus = confusionmat(test_labels, predict_labels);
        acc(k) = sum(diag(confus)) / sum(confus(:)) * 100;
    end
    features1 = [];
    features2 = [];
    performance = [performance, mean(acc)];
    plot(performance, '-o','linewidth', 2)
    hold on
    drawnow
end
end

