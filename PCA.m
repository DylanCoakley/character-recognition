function [] = PCA();

    % Load the balanced training data
    load C:\Users\Dylan\Desktop\EMINST\digTrainX.mat digTrainX
    load C:\Users\Dylan\Desktop\EMINST\digTrainY.mat digTrainY
    load C:\Users\Dylan\Desktop\EMINST\digTestX.mat digitTestX
    load C:\Users\Dylan\Desktop\EMINST\digTestY.mat digitTestY
    
    reducedTrainX = digTrainX; 
    reducedTrainY = digTrainY;
    reducedTestX = digitTestX;
    reducedTestY = digitTestY;
    
    numTrainSamples = 10000;
    
    numTestSamples = 5000;
    
    maxPCs = 100;
    totalSamples = double([reducedTrainX; reducedTestX]);
    [rotation, PCSpaceData, eigenvalues] = pca(totalSamples);
    
    for numPCs = 1 : maxPCs
        X(numPCs) = numPCs;
        Y(numPCs) = sum(eigenvalues(1:numPCs)) / sum(eigenvalues);
        
        fprintf(" With %d PCs, the variance captured is %f \n", numPCs, Y(numPCs));
    end
    
    figure;
    plot(X, Y);
    xlabel('Number of Principal Components');
    ylabel('Variance Captured');
    
    return;