function [] = ToImages()

%     % Load the testing data
%     load C:\Users\Dylan\Desktop\EMINST\emnist-digits.mat dataset
%   
%     test = dataset.('test');
%     testX = test.('images');
%     testY = test.('labels');
%     
%     % Load the balanced 1000 sample per class digit training data
%     load C:\Users\Dylan\Desktop\EMINST\digTrainX.mat digTrainX
%     load C:\Users\Dylan\Desktop\EMINST\digTrainY.mat digTrainY
%     
%     trainX = digTrainX; 
%     trainY = digTrainY;
%     
%     numTrainSamples = 10000;
%     
%     numTestSamples = 5000;
% 
%     reducedTestX = double(testX(1:numTestSamples, :));
%     reducedTestY = double(testY(1:numTestSamples, :));
%     
%     totalSamples = [trainX; reducedTestX];
    
    %reshapedSamples = nan(15000, 28, 28, 15000);
    
%     for i = 1 : size(totalSamples, 1)
%        reshapedSamples(i, 1:28, 1:28, 1) = reshape(totalSamples(i, 1:784), [28, 28]); 
%     end
    
    load C:\Users\Dylan\Desktop\EMINST\reshapedSamples2.mat reshapedSamples
    
    for i = 1 : 15000
       images(i, 1:28, 1:28, 1) = mat2gray(reshapedSamples(i), [0, 255]);
       if i == 1
           imagesc(images(1));
       end
    end


    return;