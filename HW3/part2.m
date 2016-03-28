clc;clear;

%Part 2, Adaboost + Bayes Classfier
load cancer.mat
train_size = 500;
test_size = 183;
feature_size = 9;
test_y = label(1,1:test_size); % 1 x 183
test_x = X(:,1:test_size);      % 10 x 183
train_y = label(1,test_size + 1:end);   % 1 x 500
train_x = X(:,test_size + 1:end); % 10 x 500
%Initialize
T = 1000;   % iteration number
p = zeros (1,train_size,T); % sampling distribution 1 x 500 x 1000
% Initialize sampling distribution of 1st iteration as uniform distribution
p(:,:,1) = ones(1,train_size) / train_size; 
training_error = zeros (1, T); % 1 x 1000
testing_error = zeros (1, T);   % 1 x 1000
epsilon = zeros (1, T); % 1 x 1000, 
% performs the single classifier ft (of current iteration t) 
% on original training set (before bootstrap)
alpha = zeros (1, T);   % 1 x 1000
% concatenate w0 (scaler) and w1 (9 x 1 vector) to w (10 x 1) for each
% iteration
w = zeros(feature_size + 1, T); % 10 x 1000
for t = 1 : T
    % Sample a bootstrap data set of size 500
    B_index = sample(train_size,p(:,:,t));
    B_x = train_x(:,B_index);
    B_x = B_x(2:end,:); % 9 x 500
    % B_x will not contain bias dimension 1,
    % So we can use B_x to calculate mu1, mu0 and Sigma
    B_y = train_y(:,B_index); % 1 x 500
    pi0 = sum(-1 == B_y) / train_size; % label -1 probability
    pi1 = sum(1 == B_y) / train_size;  % label +1 probability
    x0 = B_x(:,-1 == B_y); % 9 x # data point with label -1
    x1 = B_x(:,1 == B_y); % 9 x # data point with label +1
    mu0 = mean(x0,2);   % mu estimate of label -1: 9 x 1
    mu1 = mean(x1,2);   % mu estimate of label +1: 9 x 1
    % shared covariance matrix 9 x 9
    % Sigma = cov(B_x');
    Sigma = 1/ train_size * ((x1-repmat(mu1,1,size(x1,2)))*(x1-repmat(mu1,1,size(x1,2)))' + (x0-repmat(mu0,1,size(x0,2)))*(x0-repmat(mu0,1,size(x0,2)))');
    % calculate w0 and w1
    w0 = log(pi1/pi0) - 1/2 * (mu1 + mu0)' * pinv(Sigma) * (mu1 - mu0); % 1 x 1 - 1x9 * 9x9 * 9x1 = 1x1
    w1 = pinv(Sigma) * (mu1 - mu0); % 9x9 * 9x1 = 9x1
    w(:,t) = [w0 ; w1]; % 10x1
    % get current data distribution probability
    p_cur = p(:,:,t);
    error_index = sign (train_x' * [w0; w1]) ~= train_y';
    error_p = p_cur(error_index); % p_cur (500x10 * 10x1 ~= 500x1), misclassified data probability
    epsilon(1,t) = sum(error_p); % sum up the misclassified data probability and get epsilon
    alpha(1,t) = log((1-epsilon(1,t))/epsilon(1,t)) / 2; 
    % update weights    
    data_pred = train_y .* sign(train_x' * [w0; w1])'; % for 500 data point, if classify right -> 1, if wrong -> -1
    find(data_pred - 1);
    p_update = p(:,:,t) .* exp(-alpha(1,t)*data_pred ); % 1x500 .* (1x500 .* (500x10 * 10x1)'))
    if t < T
        p(:,:,t + 1) = p_update / sum(p_update); % normalize
        p_new = p(:,:,t + 1);
    end
    error_index_find = find(error_index);
%     disp(error_p);
%     disp(p_new(error_index));
    % Calculate training error
    predict = zeros(train_size,1);
    for tt = 1 : t
        predict = predict + alpha(1,tt)* sign ( (train_x' * w(:,tt))); % 500x1 + 1x1 * 500x10 * 10x1 = 500x1
    end
    training_error(1,t) = sum(sign (predict) ~= train_y') / train_size; % 500x1 ~= 500x1 => 500x1
    error_index_find_train = find(sign (predict) ~= train_y');
    % Calculate testing error
    test_predict = zeros(test_size,1);
    for tt = 1 : t
        test_predict = test_predict + alpha(1,tt) *sign( (test_x' * w(:,tt))); % 183x1 + 1x1 * 183x10 * 10x1 = 183x1
        %test_predict = test_predict + alpha(1,tt) * (w(1,tt) + test_x(2:end,:)' * w(2:end,tt));
    end
    testing_error(1,t) = sum(sign (test_predict) ~= test_y') / test_size;
    error_index_find_test = find(sign (test_predict) ~= test_y');
end
x = 1:1:1000;
figure
title('Training and testing error on iteration');
plot(x,training_error,x,testing_error);
legend('training error','testing error');
figure
plot(alpha);
figure
plot(epsilon);
fprintf(['Testing accuracy for Bayes classfier with Adaboost is ' num2str(1 - testing_error(1,T))]);

% single bayes classifier
x0 = train_x(:,-1 == train_y); % 9 x # data point with label -1
x1 = train_x(:,1 == train_y); % 9 x # data point with label +1
x0 = x0(2:end,:);
x1 = x1(2:end,:);
mu0 = mean(x0,2);   % mu estimate of label -1: 9 x 1
mu1 = mean(x1,2);   % mu estimate of label +1: 9 x 1
pi0 = sum(-1 == train_y) / train_size; % label -1 probability
pi1 = sum(1 == train_y) / train_size;  % label +1 probability
Sigma = 1/ train_size * ((x1-repmat(mu1,1,size(x1,2)))*(x1-repmat(mu1,1,size(x1,2)))' + (x0-repmat(mu0,1,size(x0,2)))*(x0-repmat(mu0,1,size(x0,2)))');
w0 = log(pi1/pi0) - 1/2 * (mu1 + mu0)' * pinv(Sigma) * (mu1 - mu0); % 1 x 1 - 1x9 * 9x9 * 9x1 = 1x1
w1 = pinv(Sigma) * (mu1 - mu0); % 9x9 * 9x1 = 9x1
w = [w0;w1];
error_ind = sign (test_x' * w) ~= test_y';
accuracy_no_boost = 1 - sum(error_ind) / test_size;
fprintf(['Testing accuracy for Bayes classfier without Adaboost is ' num2str(accuracy_no_boost)]);

% trace 3 data points on their corresponding p in terms of t
datapoint = [82,180,342];
figure
title('Trace 3 data points on their corresponding p in terms of t');
plot(x,reshape(p(1,datapoint(1),:),1,T),x,reshape(p(1,datapoint(2),:),1,T),x,reshape(p(1,datapoint(3),:),1,T));
legend(['Data ' num2str(datapoint(1))],['Data ' num2str(datapoint(2))],['Data ' num2str(datapoint(3))]);