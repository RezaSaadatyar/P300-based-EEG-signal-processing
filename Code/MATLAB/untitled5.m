clc; clear; close all;

% Randomly generated time series data
% X = rand(1000, 10);
X = rand(240, 1140)';
numObservations = size(X, 1);  % total number of sequences
sequenceLength = size(X, 2);   % length of each sequence
numClasses = 2;
% Ensure Y is a categorical array with the same number of elements as there are sequences in X
Y = categorical(randi([0, 1], numObservations, 1));

% Reshape X for CNN input: from [numObservations, sequenceLength] to [sequenceLength, 1, 1, numObservations]
X = permute(X, [2, 3, 1]);
X = reshape(X, [sequenceLength, 1, 1, numObservations]);

layers = [
    imageInputLayer([sequenceLength 1 1], 'Name', 'input', 'Normalization', 'none')

    convolution2dLayer([5, 1], 8, 'Name', 'conv1', 'Padding', 'same')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    
    maxPooling2dLayer([2, 1], 'Name', 'maxpool1', 'Stride', [2, 1])

    convolution2dLayer([5, 1], 16, 'Name', 'conv2', 'Padding', 'same')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    maxPooling2dLayer([2, 1], 'Name', 'maxpool2', 'Stride', [2, 1])
    
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

lgraph = layerGraph(layers);

options = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.01, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X, Y}, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(X, Y, lgraph, options);
y = rand(240, 1);
y = reshape(y, [240, 1, 1]);
% y = reshape(y, [numObservations, 1 ,1,1]);
YPred = classify(net, y);
accuracy = sum(YPred == Y) / numel(Y);
disp(['Validation accuracy: ', num2str(accuracy)]);