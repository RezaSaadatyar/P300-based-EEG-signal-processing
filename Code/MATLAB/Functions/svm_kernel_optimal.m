function result = svm_kernel_optimal(data, labels, k_fold, c, sigma)
% ================================== (2023-2024) ======================================
% ======================== Presented by: Reza Saadatyar ===============================
% ====================== E-mail: Reza.Saadatyar@outlook.com ===========================
% Example:
% k_fold = 5;
% c = [0.1 1 10 15 50 100];
% sigma = [1 5 10 50 80 100 120];
% svm_kernel_optimal(data, labels, k_fold, c, sigma)
% Combine target & non target data
% data = [target_data, non_target_data];
% labels = [ones(1, size(target_data, 2)), 2 * ones(1, size(non_target_data, 2))];

% Inputs:
%   - features: Features matrix.
%   - labels: Label vector.
%   - k_fold: Number of folds for cross-validation.
%   - kernel_svm: Kernel function for SVM.
%% ===================== Flowchart for the svm_optimal function =======================
% Start
% 1. Preprocess the data:
%    a. Transpose labels if necessary.
%    b. Create k-fold cross-validation indices using 'cvpartition'.
%    c. Initialize the result matrix to store parameter combinations and accuracies.
% 2. Iterate over each combination of C and sigma values:
%    a. Iterate over the specified range of C values.
%    b. Iterate over the specified range of sigma values.
%    c. Perform k-fold cross-validation:
%       - Split the data into training and test sets.
%       - Train an SVM model using 'fitcsvm'.
%       - Predict labels for the test set.
%       - Calculate accuracy using confusion matrix.
%    d. Update the result matrix with the mean accuracy for the current parameter combination.
%    e. Print the current parameter combination and its accuracy.
% 3. Find the optimal combination of C and sigma based on the highest mean accuracy.
% 4. Print the optimal C and sigma values along with their accuracy
% End
%% ====================================================================================
% Data preprocessing
if size(labels, 1) < size(labels, 2); labels = labels'; if size(labels, 2) > 1
        labels = vec2ind(labels'); end

    % if size(features, 1) < size(features, 2); features = features'; end
    count = 0;
    indx = cvpartition(labels, 'k', k_fold);      % k-fold cross validation
    result = zeros(length(c) * length(sigma), 3);
    for i = 1:length(c)
        for j = 1:length(sigma)
            for k = 1:k_fold  % Perform k-fold cross-validation
                train_ind = indx.training(k); test_ind = indx.test(k);
                train_data = data(:, train_ind); train_labels = labels(train_ind);
                test_data = data(:, test_ind); test_labels = labels(test_ind);
                % ---------------------------------- SVM ---------------------------------
                mdl= fitcsvm(train_data', train_labels, 'Standardize', 1, 'KernelFunction', ...
                    'rbf', 'KernelScale', sigma(j), 'BoxConstraint', c(i), 'Verbose', 0);
                predict_labels = predict(mdl, test_data');
                confus = confusionmat(test_labels, predict_labels);
                acc(k) = sum(diag(confus)) / sum(confus(:)) * 100;

            end
            count = count + 1;
            result(count, :) = [c(i), sigma(j), mean(acc)];
            fprintf('C: %s, Sigma: %s --> Accuracy: %.2f\n', num2str(c(i)), num2str(sigma(j)),...
                mean(acc));
        end
    end
    [~, ind] = max(result(:, 3));
    fprintf('C_optimal: %s, Sigma_optimal: %s --> Accuracy: %.2f\n', num2str(result(ind, 1)), ...
        num2str(result(ind, 2)), result(ind, 3));
end