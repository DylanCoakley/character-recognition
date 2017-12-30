function [] = SVM()

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
    
    % Create the linear kernel multi-class SVM model
    svmTemplate = templateSVM('KernelFunction', 'linear', ...
            'KernelScale', 'auto', ...
            'Standardize', 1);
    model = fitcecoc(pcaTrainX, trainY, 'Learners', svmTemplate);
    [modelPrediction] = predict(model, pcaTestX);
    
    accuracy = sum(testY == modelPrediction) / numel(testY);

    fprintf("Linear-kernel SVM has accuracy of %f \n", accuracy);
    
    
    for d = 2 : 5
        % Create the polynomial kernel multi-class SVM model
        svmTemplate = templateSVM('KernelFunction', 'polynomial', ...
            'PolynomialOrder', d, ...
            'KernelScale', 'auto', ...
            'Standardize', 1);
        model = fitcecoc(pcaTrainX, trainY, 'Learners', svmTemplate);
        [modelPrediction] = predict(model, pcaTestX);

        accuracy = sum(testY == modelPrediction) / numel(testY);

        fprintf("Polynomial-kernel SVM of degree %d has accuracy of %f \n", d, accuracy);
    end
    
    % Create multi-class SVM model using RBF
    svmTemplate = templateSVM('KernelFunction', 'rbf', ...
        'KernelScale', 'auto', ...
        'Standardize', 1);
    model = fitcecoc(pcaTrainX, trainY, 'Learners', svmTemplate);
    [modelPrediction] = predict(model, pcaTestX);

    accuracy = sum(testY == modelPrediction) / numel(testY);

    fprintf("RBF-kernel SVM has accuracy of %f \n", accuracy);
   
    
     return;
    
        