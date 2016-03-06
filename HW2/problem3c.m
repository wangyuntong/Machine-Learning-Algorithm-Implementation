%load data
load mnist_mat.mat
[featureNum,trainSize] = size(Xtrain);
testSize = size(Xtest,2);
classNum = 10;
%initialize  
w = zeros(featureNum + 1, classNum); % 21 x 10
L = 0;
X = [Xtrain;ones(1,trainSize)]; % 21 x 5000
stepSize = 0.1/5000;
iterations = 1000;
cost = zeros(iterations,1);

% train
for i = 1:iterations
    [cost(i,:),delta] = costFunctionReg(X,label_train,w);
    w = w + stepSize * delta;
end

%predict
[maxValue,predIdx] = max([Xtest;ones(1,testSize)]'*w,[],2);
prediction = predIdx - 1;


figure;
title('cost');
plot(cost);

%find confusion matrix and accurancy for each k
C = confusionmat(label_test,prediction);
accurancy = trace(C)/500;
fprintf('The accurancy for multiclass logistic regression is : %.4f\n', accurancy);

%show 3 misclassified images
%show the probabaility distribution on 10 digits learned by softmax
%function of these 3 misclassified images
misclassifiedIndex = find(prediction' - label_test);
figure
axis image
title('misClassfied examples on softmax');
for i = 1:3
    subplot(1,3,i);
    imshow(reshape((Q*Xtest(:,misclassifiedIndex(i))), [28,28])');
    title(['prediction: ' , num2str(prediction(misclassifiedIndex(i))), ' ground truth: ',num2str(label_test(misclassifiedIndex(i)))]);
    fprintf('the probabaility distribution on 10 digits for misclassified image %d: \n',i);
    disp([Xtest(:,misclassifiedIndex(i));1]'* w);
end






