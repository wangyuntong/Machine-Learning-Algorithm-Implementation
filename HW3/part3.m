clc;clear;
load cancer.mat
train_size = 500;
test_size = 183;
feature_size = 10;
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
step = 0.1;
epsilon = zeros (1, T); % 1 x 1000, 
% performs the single classifier ft (of current iteration t) 
% on original training set (before bootstrap)
alpha = zeros (1, T);   % 1 x 1000
w_it = zeros(feature_size,T);
for t = 1 : T
    w = zeros(feature_size,1); % 10 x 1
    % Sample a bootstrap data set of size 500
    B_index = sample(train_size,p(:,:,t));
    B_x = train_x(:,B_index); % 10x500z
    B_y = train_y(:,B_index); % 1 x 500
    p_cur = p(:,:,t);
    for i = 1 : train_size
        sigma = 1/(1 + exp(-B_y(1,i) * B_x(:,i)' * w));
        w = w + step * B_x(:,i) * B_y(1,i) * (1 - sigma); % 10x1
    end
    w_it(:,t) = w;
    error_index = sign(train_x' * w) ~= train_y';
    error_p = p_cur(error_index); % p_cur (500x10 * 10x1 ~= 500x1), misclassified data probability
    epsilon(1,t) = sum(error_p); % sum up the misclassified data probability and get epsilon
    alpha(1,t) = log((1-epsilon(1,t))/epsilon(1,t)) / 2; 
    % update weights
    data_pred = train_y .* sign(train_x' * w)'; % for 500 data point, if classify right -> 1, if wrong -> -1
    find(data_pred - 1);
    p_update = p(:,:,t) .* exp(-alpha(1,t)*data_pred ); % 1x500 .* (1x500 .* (500x10 * 10x1)'))
    if t < T
        p(:,:,t + 1) = p_update / sum(p_update); % normalize
    end
    % Calculate training error
    predict = zeros(train_size,1);
    for tt = 1 : t
        predict = predict + alpha(1,tt) * sign(train_x' * w_it(:,tt)); % 500x1 + 1x1 * 500x10 * 10x1 = 500x1
    end
    training_error(1,t) = sum(sign(predict) ~= train_y') / train_size;
    
    % Calculate testing error
    test_predict = zeros(test_size,1);
    for tt = 1 : t
        test_predict = test_predict + alpha(1,tt) * sign (test_x' * w_it(:,tt)); % 183x1 + 1x1 * 183x10 * 10x1 = 183x1
        %test_predict = test_predict + alpha(1,tt) * (w(1,tt) + test_x(2:end,:)' * w(2:end,tt));
    end
    testing_error(1,t) = sum(sign(test_predict) ~= test_y') / test_size;
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

% logistic regression without adaboost
w = zeros(feature_size,1); % 10 x 1
for i = 1 : train_size
    sigma = 1/(1 + exp(-train_y(1,i) * train_x(:,i)' * w));
    w = w + step * train_x(:,i) * train_y(1,i) * (1 - sigma); % 10x1
end
error_ind = sign(test_x' * w) ~= test_y';
sum(error_ind);
accuracy_no_adaboost = 1 - sum(error_ind) / test_size;
fprintf(['Testing accuracy of binary logistic regression without Adaboost is ' num2str(accuracy_no_adaboost)]);
% trace 3 data points on their corresponding p in terms of t
datapoint = [12,438,493];
figure
title('Trace 3 data points on their corresponding p in terms of t');
plot(x,reshape(p(1,datapoint(1),:),1,T),x,reshape(p(1,datapoint(2),:),1,T),x,reshape(p(1,datapoint(3),:),1,T));
legend(['Data ' num2str(datapoint(1))],['Data ' num2str(datapoint(2))],['Data ' num2str(datapoint(3))]);