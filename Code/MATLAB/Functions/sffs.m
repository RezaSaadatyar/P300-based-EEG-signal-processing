function [optimal_channels, optimal_perfomance] = sffs(data1, data2, max_channel)
% ================================== (2023-2024) ======================================
% ======================== Presented by: Reza Saadatyar ===============================
% ====================== E-mail: Reza.Saadatyar@outlook.com ===========================
% Example:
% load data
% max_channel = 10;
% removed_channels = sffs(target_data, non_target_data, max_channel);
% ========================= Flowchart for the SFFS function ===========================
% 1. Start 
% 2. Initialize select_channel as all available channels.
% 3. Initialize an empty array to store performance results (performance).
% 4. Initialize an empty array to store optimal selected channels (optimal_channels).
% 5. Iterate over the range of max_channel:
%    a. Iterate over each channel in select_channel:
%       i. Append the current channel to optimal_channels.
%       ii. Extract features from data1 and data2 based on the current optimal_channels.
%       iii. Concatenate features from both datasets.
%       iv. Split data into training and test sets.
%       v. Train an SVM model using 'fitcsvm' with the training data and labels.
%       vi. Predict labels for the test set.
%       vii. Calculate accuracy using the confusion matrix.
%       viii. Store the accuracy in the performance array.
%    b. Sort the performance array in descending order and select the highest accuracy.
%    c. Store the optimal performance and corresponding channel in optimal_perfomance 
%       and optimal_channels, respectively.
%    d. Remove the selected channel from select_channel.
%    e. Clear the feature arrays.
% 6. End the function.
% =====================================================================================
select_channel = 1:size(data1, 2);
perfomance = [];
optimal_channels = [];

for epoch = 1:max_channel

    for ind_ch = select_channel

        ind_optim_ch = [optimal_channels, ind_ch];
        % ----------------------------- feature extaction -----------------------------
        for i = 1:size(data2, 3)
            x1 = data1(:, ind_optim_ch, i);
            x2 = data2(:, ind_optim_ch, i);
            
            features_1(:, i) = x1(:);
            features_2(:, i) = x2(:);
        end
        % ----------------- Split data into training and test data --------------------
        split_data= round(0.7 * size(features_1, 2));
        train_data = [features_1(:, (1:split_data)), features_2(:, (1:split_data))];
        test_data = [features_1(:, (split_data + 1:end)), features_2(:, (split_data + 1:end))];
        train_labels = [ones(1, size(features_1(:, (1:split_data)), 2)), ... 
                        2 * ones(1, size(features_2(:, (1:split_data)), 2))];
        test_labels = [ones(1, size(features_1(:, (split_data + 1:end)), 2)), ...
                       2 * ones(1, size(features_2(:, (split_data + 1:end)), 2))];
        % ----------- Train & test classifier using train data & train label ----------
        model = fitcsvm(train_data', train_labels, 'Standardize', 1, 'BoxConstraint', 0.01);
        predict_labels = predict(model, test_data'); % test trained model using test data
        conf = confusionmat(test_labels, predict_labels);
        accuracy = sum(diag(conf)) / sum(conf(:)) * 100;
        perfomance = [perfomance, accuracy];
    end
    [perfomance, ind] = sort(perfomance, 'descend');
    optimal_perfomance(epoch) = perfomance(1);
    optimal_channels = [optimal_channels, select_channel(ind(1))];
    perfomance = [];
    features_1 = [];
    features_2 = [];
    select_channel(ind(1)) = [];
end
