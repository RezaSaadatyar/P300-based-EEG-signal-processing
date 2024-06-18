clc;clear; close all
% Parameters
numSamples = 240; % Number of samples
numChannels = 570; % Number of channels
numClasses = 50; % Number of classes
imageSize = [numSamples, numChannels, 1];

% Generate synthetic EEG data
numObservationsPerClass = 10;

class1Data = zeros([numSamples, numChannels, 1, numObservationsPerClass]);
class2Data = zeros([numSamples, numChannels, 1, numObservationsPerClass]);

% Class 1: Random noise with a specific pattern
for i = 1:numObservationsPerClass
    data = randn(numSamples, numChannels);
    class1Data(:,:,:,i) = data;
end

% Class 2: Different random noise with another pattern
for i = 1:numObservationsPerClass
    data = randn(numSamples, numChannels);
    class2Data(:,:,:,i) = data;
end

% Combine the data
X = cat(4, class1Data, class2Data);
Y = categorical([ones(numObservationsPerClass, 1); 2*ones(numObservationsPerClass, 1)]);

% Split data into training and validation sets
numTrain = 0.8 * 2 * numObservationsPerClass;
idx = randperm(2 * numObservationsPerClass);

XTrain = X(:,:,:,idx(1:numTrain));
YTrain = Y(idx(1:numTrain));

XValidation = X(:,:,:,idx(numTrain+1:end));
YValidation = Y(idx(numTrain+1:end));

% Define the CNN architecture
layers = [
    imageInputLayer(imageSize)
    convolution2dLayer([3 3], 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride', [2 2])
    convolution2dLayer([3 3], 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride', [2 2])
    convolution2dLayer([3 3], 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Set training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {XValidation, YValidation}, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');


% Train the CNN
net = trainNetwork(XTrain, YTrain, layers, options);

% Evaluate the trained network
YPred = classify(net, XValidation);
accuracy = sum(YPred == YValidation) / numel(YValidation);
disp(['Validation accuracy: ', num2str(accuracy)]);
