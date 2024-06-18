clc; clear; close all

% Generate synthetic data for two classes
numFeatures = 570; % Number of features per sample
numSamples = 240; % Number of samples per class
classes = 2; % Number of classes

% Generate random data
data = rand([numFeatures, 1, 1, numSamples * classes]);
labels = categorical([ones(1, numSamples), 2*ones(1, numSamples)]);

% Split data into training and validation sets
idx = randperm(numSamples * classes);
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

% Evaluate the network
YPred = classify(net, valData);
accuracy = sum(YPred == valLabels) / numel(valLabels);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);
