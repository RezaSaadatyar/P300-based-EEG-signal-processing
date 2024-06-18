clc;clear;
parentDir = tempdir;
cd(parentDir)
mkdir("BonnEEG")
dataDir = fullfile(parentDir,"BonnEEG");
dataDir = "/tmp/BonnEEG";
unzip("z.zip",dataDir)
unzip("o.zip",dataDir)
unzip("n.zip",dataDir)
unzip("f.zip",dataDir)
unzip("s.zip",dataDir)

netSPN = [sequenceInputLayer(1,"MinLength",4097,"Name","input","Normalization","zscore")
    convolution1dLayer(5,1,"stride",2)
    cwtLayer("SignalLength",2047,"IncludeLowpass",true,"Wavelet","amor")
    maxPooling2dLayer([5,10])
    convolution2dLayer([5,10],5,"Padding","same")
    maxPooling2dLayer([5,10])  
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([5,10],10,"Padding","same")
    maxPooling2dLayer([2,4])   
    batchNormalizationLayer
    reluLayer
    flattenLayer
    globalAveragePooling1dLayer
    dropoutLayer(0.4)
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer("Classes",unique(trainLabelsSPN),"ClassWeights",classwghts)
    ];

options = trainingOptions("adam", ...
    "MaxEpochs",40, ...
    "MiniBatchSize",20, ...
    "Shuffle","every-epoch",...
    "Plots","training-progress",...
    "ValidationData",{validationDataSPN,validationLabelsSPN},...
    "L2Regularization",1e-2,...
    "OutputNetwork","best-validation-loss",...
    "Verbose", false);

trainedNetSPN = trainNetwork(trainDataSPN,trainLabelsSPN,netSPN,options);