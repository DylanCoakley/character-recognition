function [] = DiscriminantAnalysis()

    % Load the balanced training data
    load C:\Users\Dylan\Desktop\EMINST\digTrainX.mat digTrainX
    load C:\Users\Dylan\Desktop\EMINST\digTrainY.mat digTrainY
    load C:\Users\Dylan\Desktop\EMINST\digTestX.mat digitTestX
    load C:\Users\Dylan\Desktop\EMINST\digTestY.mat digitTestY
    
    trainX = digTrainX; 
    trainY = digTrainY;
    testX = digitTestX;
    testY = digitTestY;
    
    numTrainSamples = 10000;
    
    numTestSamples = 5000;
  
    % Perform PCA with 75 PCs
    numPCs = 75;
    totalSamples = [trainX; testX];
    [rotation, PCSpaceData, eigenvalues] = pca(double(totalSamples));
    pcaTrainX = PCSpaceData(1:numTrainSamples,1:numPCs);
    pcaTestX = PCSpaceData(numTrainSamples+1:size(totalSamples),1:numPCs);
    fprintf("Number of features reduced to %d using PCA. \n", numPCs);
    
    % Create the LDA multi-class model
    model = fitcdiscr(pcaTrainX, trainY, 'DiscrimType', 'pseudolinear');%, ...
        %'OptimizeHyperparameters', {'Delta','Gamma'});
    [modelPrediction] = predict(model, pcaTestX);
    
    accuracy = sum(testY == modelPrediction) / numel(testY);

    fprintf("Linear discriminant analysis has accuracy of %f \n", accuracy);
    
    % Create the QDA multi-class model
    model = fitcdiscr(pcaTrainX, trainY, 'DiscrimType', 'pseudoquadratic');%, ...
        %'OptimizeHyperparameters', {'Delta','Gamma'});
    [modelPrediction] = predict(model, pcaTestX);
    
    accuracy = sum(testY == modelPrediction) / numel(testY);

    fprintf("Quadratic discriminant analysis has accuracy of %f \n", accuracy);

    return;