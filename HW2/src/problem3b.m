%load data
load mnist_mat.mat
[featureNum,trainSize] = size(Xtrain);
testSize = size(Xtest,2);
classNum = 10;

% Calculate the class prior for MLE estimate
classN = zeros(classNum,1); % 10 x 1
for i = 1 : trainSize
    classN(label_train(:,i) + 1,:) = classN(label_train(:,i) + 1,:) + 1;
end
classPrior = classN / trainSize;
for j = 0 : classNum - 1
   fprintf('Class Prior for %d is %.2f\n',j, classPrior(j + 1));
end

%MLE for Multivariate Gaussian distribution , all 10 x 20
mu = zeros(classNum, featureNum);
Sigma = zeros(classNum, featureNum);

for j = 1 : classNum
    mu(j,:) = sum(repmat((label_train + 1)==j,featureNum,1).*Xtrain, 2)'/classN(j,1);
    Sigma(j,:) = sum((repmat((label_train + 1)==j,featureNum,1).*(Xtrain - repmat(mu(j,:)',1,trainSize))).^2, 2)'/classN(j,1);
end
fprintf('MLE of mean:\n');
disp(mu);
fprintf('MLE of covariance:\n');
disp(Sigma);

%predict
Prob = ones(classNum, testSize);
for j = 1 : classNum
    for i = 1 : testSize
        for t = 1 : featureNum
            Prob(j,i) = Prob(j,i) * classPrior(j) * abs(Sigma(j,t)).^(-0.5) * exp((-1/2)*(Xtest(t,i) - mu(j,t)).^2 / Sigma(j,t));
        end
    end
end

[maxValue,predIdx]  = max(Prob,[],1);
prediction = predIdx - 1;

%find confusion matrix and accurancy for each k
C = confusionmat(label_test,prediction);
accurancy = trace(C)/500;
fprintf('The accurancy for Bayes Classifier is : %.4f\n', accurancy);

%Show the mean of each Gaussian as an image using the provided Q matrix
figure;
for i = 1 : classNum
    subplot(2,5,i)
    imshow(reshape((Q*mu(i,:)'), [28,28])');
    title(['digit ',num2str(i-1)]);
end

%show misclassified images
misclassifiedIndex = find(prediction - label_test);
figure
axis image
title('misClassfied examples on softmax');
for i = 1:3
    subplot(1,3,i);
    imshow(reshape((Q*Xtest(:,misclassifiedIndex(i))), [28,28])');
    title(['prediction: ' , num2str(prediction(misclassifiedIndex(i))), ' ground truth: ',num2str(label_test(misclassifiedIndex(i)))]);
    fprintf('the probabaility distribution on 10 digits for misclassified image %d: \n',i);
    disp(Prob(:,misclassifiedIndex(i))');
end


