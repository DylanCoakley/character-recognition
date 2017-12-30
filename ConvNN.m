function [] = ConvNN()

    trainFolder = fullfile('C:', 'Users', 'Dylan', 'Desktop', 'EMINST', ...
        'emnist_digit_train_images');
    categories = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    imdsTrain = imageDatastore(fullfile(trainFolder, categories), ...
         'LabelSource', 'foldernames');
%     imdsTrain = splitEachLabel(imdsTrain, 1000, 'randomize');
    tbl = countEachLabel(imdsTrain)
%     
    testFolder = fullfile('C:', 'Users', 'Dylan', 'Desktop', 'EMINST', ...
        'emnist_digit_test_images');
    imdsTest = imageDatastore(fullfile(testFolder, categories), ...
        'LabelSource', 'foldernames');
%     imdsTest = splitEachLabel(imdsTest, 500, 'randomize');
    tbl = countEachLabel(imdsTest)
    
%     layers = [ ...
%         imageInputLayer([28 28 1]); % 28x28 with 1 channel (grayscale)
%         convolution2dLayer(12, 25); % a very basic CNN with a single convolutional layer
%         reluLayer();
%         fullyConnectedLayer(10); % connects all features in previous layer to all nodes 
%         softmaxLayer();
%         classificationLayer(); ...
%     ];

    layers = [
        imageInputLayer([28 28 1])  
        convolution2dLayer(3,16,'Padding',1)
        batchNormalizationLayer
        reluLayer    
        maxPooling2dLayer(2,'Stride',2) 
        convolution2dLayer(3,32,'Padding',1)
        batchNormalizationLayer
        reluLayer 
        fullyConnectedLayer(10)
        softmaxLayer
        classificationLayer];

    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.0001, ...
        'MaxEpochs', 15, ...
        'Plots', 'training-progress'); % faster training if you lower MaxEpochs, may produce weaker results
    
    net = trainNetwork(imdsTrain, layers, options)
    modelY = classify(net, imdsTest);
    testY = imdsTest.Labels;
    
    %accuracy = sum(modelPrediction == reducedTestY) / numel(reducedTestY);
    accuracy = sum(modelY == testY) / numel(testY);

    fprintf("Convolutional Neural Network has accuracy of %f \n", accuracy);

    return;