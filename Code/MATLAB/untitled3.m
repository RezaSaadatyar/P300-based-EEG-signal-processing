clc; clear; close all;


% Generate synthetic data for two classes
imageSize = [28, 28, 1]; % Image dimensions
numImages = 500; % Number of images per class
classes = 2; % Number of classes

% Generate random images
data = rand([imageSize, numImages, classes]);
labels = categorical([ones(1, numImages), 2*ones(1, numImages)]);

% Split data into training and validation sets
idx = randperm(numImages * classes);
trainData = data(:, :, :, idx(1:round(0.8*numImages*classes)));
trainLabels = labels(idx(1:round(0.8*numImages*classes)));
valData = data(:, :, :, idx(round(0.8*numImages*classes)+1:end));
valLabels = labels(idx(round(0.8*numImages*classes)+1:end));

% Define the CNN architecture
layers = [
    imageInputLayer(imageSize)
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
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
