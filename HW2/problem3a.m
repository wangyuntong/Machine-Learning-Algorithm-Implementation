%load data
clc;clear;close all;
load mnist_mat.mat
[featureNum,trainSize] = size(Xtrain);
testSize = size(Xtest,2);
testK = 5;
% train KNN: calculate the distance between test data and train data
index = zeros(testSize,trainSize);
nnlabel = zeros(testSize, testK);
for i = 1:testSize
    distance = sum((repmat(Xtest(:,i),1,trainSize) - Xtrain).^2,1).^0.5;
    [distance,index(i,:)] = sort(distance);
    nnlabel(i,:) = label_train(:,index(i,1:testK));
end

%classfy using KNN
prediction = zeros(testK, testSize);
for k = 1:testK
    for i = 1:testSize
        table = tabulate(nnlabel(i,1:k));
        [maxCount,idx] = max(table(:,2));
        prediction(k,i) = table(idx);
    end
end

%find confusion matrix and accurancy for each k
C = zeros(10, 10, testK);
accurancy = zeros(1,testK);
for k = 1:testK
    C(:,:,k) = confusionmat(label_test,prediction(k,:));
    accurancy(:,k) = trace(C(:,:,k))/500;
    fprintf('The accurancy for k-NN with %d neighbors : %.4f\n', k, accurancy(:,k));
end

%show misclassfied examples
for k = [1,3,5]
    misclassifiedIndex = find(prediction(k,:) - label_test);
    figure
    axis image
    for i = 1:3
        subplot(1,3,i);
        imshow(reshape((Q*Xtest(:,misclassifiedIndex(i))), [28,28])');
        title(['prediction: ' , num2str(prediction(k,misclassifiedIndex(i))), ' ground truth: ',num2str(label_test(misclassifiedIndex(i)))]);
    end
end

    
    

    
        