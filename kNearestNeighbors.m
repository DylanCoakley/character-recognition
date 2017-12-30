function [overallAccuracy] = kNearestNeighbors();

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
    
    maxNeighbours = 13;
    
    % Create kNN models with different values of k
    for k = 1 : 2 : maxNeighbours
        
        pcaModel = fitcknn(pcaTrainX, trainY, 'NumNeighbors', k);
        [pcaModelPrediction] = predict(pcaModel, pcaTestX);
        
        accuracy = sum(testY == pcaModelPrediction) / numel(testY);

        fprintf("kNN with k = %d w/ %d PCs has accuracy of %f \n", k, numPCs, accuracy);
          
    end
    
    % Create and plot a confusion matrix
    n = numTestSamples;
    isLabels = unique(trainY);
    nLabels = numel(isLabels);
    oofLabel = pcaModelPrediction;
    
    % Convert the integer label vector to a class-identifier matrix.
    [~,grpOOF] = ismember(oofLabel,isLabels); 
    oofLabelMat = zeros(nLabels,n); 
    idxLinear = sub2ind([nLabels n],grpOOF,(1:n)'); 
    oofLabelMat(idxLinear) = 1; % Flags the row corresponding to the class 
    [~,grpY] = ismember(testY,isLabels); 
    YMat = zeros(nLabels,n); 
    idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
    YMat(idxLinearY) = 1; 

    figure;
    plotconfusion(YMat,oofLabelMat);
    h = gca;
    h.XTickLabel = [num2cell(isLabels); {''}];
    h.YTickLabel = [num2cell(isLabels); {''}];
    xlabel('Actual Digit');
    ylabel('Predicted Digit');

    
    return;
        