%% Solve linear regression using least square
%% Initialization
clear ; close all; clc


%% Part 1
load data;

trainSize = 372;
testSize = 20;
dataSize = size(X,1);
test_num = 1000;
for i = 1:test_num
    randInd = randperm(dataSize);
    train_Xdata = X(randInd(1:trainSize),:);
    test_Xdata = X(randInd(trainSize + 1:dataSize),:);
    train_Ydata = y(randInd(1:trainSize),:);
    test_Ydata = y(randInd(trainSize + 1:dataSize),:);  
    w(i,:) = pinv(train_Xdata)*train_Ydata;
    MAE(i) = 1 / testSize * sum( abs(test_Xdata * w(i,:)' - test_Ydata ));
end

mean_MAE = mean(MAE);
std_MAE = std(MAE);

fprintf('Statistics on the MAE for these %d tests: \n', test_num);
fprintf('Mean              : %.2f\n',mean_MAE);
fprintf('Standard deviation: %.2f\n',std_MAE);
fprintf('\n');
%% Print the numbers you obtain for the vector w with labels:
%	x1: intercept term
% 	x2: number of cylinders
% 	x3: displacement
% 	x4: horsepower
% 	x5: weight
% 	x6: acceleration
% 	x7: model year
fprintf('Example: \nParameters w obtained from linear regression using least square: \n');
fprintf('intercept term      : %.4f\n', w(i, 1));
fprintf('number of cylinders : %.4f\n', w(i, 2));
fprintf('displacement        : %.4f\n', w(i, 3));
fprintf('horsepower          : %.4f\n', w(i, 4));
fprintf('weight              : %.4f\n', w(i, 5));
fprintf('acceleration        : %.4f\n', w(i, 6));
fprintf('model year          : %.4f\n', w(i, 7));

fprintf('\n');
%% Part 2
feature_num = size(X,2);
poly_num = 4;
RMAE = zeros(poly_num,test_num);
for p = 1:poly_num
    w_poly = zeros(test_num, 1 + (feature_num - 1) * p);
    for i = 1:test_num
        randInd = randperm(dataSize);
        %initialize training data for X
        train_Xdata = X(randInd(1:trainSize),:);
        test_Xdata = X(randInd(trainSize + 1:dataSize),:);
        % Add columns to pth order polynomial regression model
        for poly = 2:p
            train_Xdata_poly = X(randInd(1:trainSize),:).^poly;
            test_Xdata_poly = X(randInd(trainSize + 1:dataSize),:).^poly;
            train_Xdata = [train_Xdata train_Xdata_poly(:, 2 : feature_num)];
            test_Xdata = [test_Xdata test_Xdata_poly(:, 2 : feature_num)];
        end
        %Initialize training data for y
        train_Ydata = y(randInd(1:trainSize),:);
        test_Ydata = y(randInd(trainSize + 1:dataSize),:);
        
        %Calculate w
        w_poly(i,:) = pinv(train_Xdata)*train_Ydata;
        %Calculate root mean absolute error for pth order polynomial
        error_new(p,i,:) = test_Ydata - test_Xdata * w_poly(i,:)';
        RMAE(p,i) =   sqrt( sum(error_new(p,i,:) .^ 2 , 3) / testSize );
        
    end
    mean_RMAE(p) = mean(RMAE(p,:));
    std_RMAE(p) = std(RMAE(p,:));

end

fprintf('p     Mean_RMAE      Std_RMAE\n\n');
for p = 1:poly_num
    fprintf('%d      %.4f         %.4f\n',p,mean_RMAE(p),std_RMAE(p));
end

%%Plot the Result
for p = 1:poly_num
    figure;
    histogram(error_new(p,:,:));
end

%%
%calculate the maximum likelihood values for the mean and variance
% Using Gaussian Model
miu_MLE = zeros(1,poly_num);
sigma_MLE = zeros(1,poly_num);

for p = 1:poly_num
    error_p = reshape( error_new(p,:,:), [ testSize * test_num , 1] );
    miu_MLE(p) = mean(error_p);
    sigma_MLE(p) = sqrt(mean((error_p - miu_MLE(p)).^2));
end
fprintf('\n');
fprintf('p     miu_MLE      Sigma_MLE\n\n');
for p = 1:poly_num
    fprintf('%d      %.4f         %.4f\n',p,miu_MLE(p),sigma_MLE(p));
end

%log likelihood
log_likelihood = zeros(1,poly_num);
for p = 1:poly_num
    error_p = reshape( error_new(p,:,:), [ testSize * test_num , 1] );
    pd = makedist('normal','mu',miu_MLE(p),'sigma',sigma_MLE(p));
    log_likelihood(p) = sum(log( pdf(pd,error_p)));
end
fprintf('\n');
fprintf('p     log likelihood\n\n');
for p = 1:poly_num
    fprintf('%d      %.4f\n',p,log_likelihood(p));
end

    
