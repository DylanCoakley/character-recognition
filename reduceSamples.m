function [] = reduceSamples();

    % Load the dataset
    load C:\Users\Dylan\Desktop\EMINST\emnist-digits.mat dataset
    
    train = dataset.('train');
    trainX = train.('images');
    trainY = train.('labels');
    test = dataset.('test');
    testX = test.('images');
    testY = test.('labels');
    
    % Make a new balanced dataset that contains 47000 total samples,
    % 1000 samples per class. This is a reduction of the number of 
    % samples in the original EMNIST-balanced dataset from 112800. 
    numClasses = 10;
    numToKeep = 1000;
    
    for i = 1 : numClasses
        indices = find(trainY == i - 1, numToKeep);
        startPos = (i - 1) * numToKeep + 1;
        endPos = i * numToKeep;
        digTrainX(startPos:endPos, 1:784) = trainX(indices, 1:784);
        digTrainY(startPos:endPos, 1:1) = trainY(indices, 1:1);
    end
    
    % Make a new balanced dataset that contains 500 samples per class
    % in the testing set
    numToKeep = 500;
    
    for i = 1 : numClasses
        indices = find(testY == i - 1, numToKeep);
        startPos = (i - 1) * numToKeep + 1;
        endPos = i * numToKeep;
        digitTestX(startPos:endPos, 1:784) = testX(indices, 1:784);
        digitTestY(startPos:endPos, 1:1) = testY(indices, 1:1);
    end

    return;