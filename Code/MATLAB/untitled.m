clc;clear;close all
% Generate synthetic data
numObservations = 1000;
imageSize = [28 28 1];

% Class 1: Generate circular patterns
class1Data = zeros([imageSize, numObservations]);
for i = 1:numObservations
    [x, y] = meshgrid(1:28, 1:28);
    circle = sqrt((x - 14).^2 + (y - 14).^2) < 10;
    class1Data(:,:,1,i) = circle;
end

% Class 2: Generate square patterns
class2Data = zeros([imageSize, numObservations]);
for i = 1:numObservations
    square = zeros(28, 28);
    square(8:20, 8:20) = 1;
    class2Data(:,:,1,i) = square;
end

% Combine the data
X = cat(4, class1Data, class2Data);
Y = categorical([ones(numObservations, 1); 2*ones(numObservations, 1)]);

% Split data into training and validation sets
numTrain = 0.8 * 2 * numObservations;
idx = randperm(2 * numObservations);

XTrain = X(:,:,:,idx(1:numTrain));
YTrain = Y(idx(1:numTrain));

XValidation = X(:,:,:,idx(numTrain+1:end));
YValidation = Y(idx(numTrain+1:end));

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
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(2)
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
