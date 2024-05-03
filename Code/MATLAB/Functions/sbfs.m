function [removed_channels, best_perfomance] = sbfs(data1, data2, max_channel)
% ================================== (2023-2024) ======================================
% ======================== Presented by: Reza Saadatyar ===============================
% ====================== E-mail: Reza.Saadatyar@outlook.com ===========================
% Example:
% load data
% max_channel = 10;
% removed_channels = sbfs(target_data, non_target_data, max_channel);
%% ======================== Flowchart for the SBFS function ===========================
% 1. Start
% 2. Initialize an empty array to store performance results (performance).
% 3. Initialize an empty array to store removed channels (removed_channels).
% 4. Initialize select_channel as all available channels.
% 
% 5. Iterate over the range of max_channel:
%    a. Iterate over each channel in select_channel:
%       i. Create a copy of select_channel and remove the current channel from it.
%       ii. Extract features from data1 and data2 based on the remaining channels.
%       iii. Concatenate features from both datasets.
%       iv. Split data into training and test sets.
%       v. Train an SVM model using 'fitcsvm' with the training data and labels.
%       vi. Predict labels for the test set.
%       vii. Calculate accuracy using the confusion matrix.
%       viii. Store the accuracy in the performance array.
%    b. Sort the performance array in descending order and select the highest accuracy.
%    c. Store the optimal performance and corresponding channels in best_perfomance and
%       removed_channels, respectively.
%    d. Remove the selected channels from select_channel.
%    e. Clear the feature arrays.
% 6. End
%% ====================================================================================
perfomance = [];
removed_channels = [];
select_channel = 1:size(data1, 2);

for epoch = 1:max_channel

    for ind_ch = select_channel
        select = select_channel;
        select(find(select==ind_ch)) = [];
        ind_optim_ch = select;
        % ----------------------------- feature extaction -----------------------------
        for i = 1:size(data2, 3)
            x1 = data1(:, ind_optim_ch, i);
            x2 = data2(:, ind_optim_ch, i);
            
            features_1(:, i) = x1(:);
            features_2(:, i) = x2(:);
        end
        % ----------------- Split data into training and test data --------------------
        split_data= round(0.5 * size(features_1, 2));
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
    best_perfomance(epoch) = perfomance(1);
    % removed_channels = [removed_channels, select_channel(ind(1))];
    removed_channels = [removed_channels, select_channel(ind(1:4))];
    perfomance = [];
    features_1 = [];
    features_2 = [];
    % select_channel(ind(1)) = [];
    select_channel(ind(1:4)) = [];
end
