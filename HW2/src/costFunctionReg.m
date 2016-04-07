function [L, grad] = costFunctionReg(X,y,w)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression
%   L = COSTFUNCTIONREG(X,y,w) computes the cost of using
%   w as the parameter for logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

%Initialize
n = size(X,2); %training examples number
classNum = size(w,2); % class number
L = 0;
grad = zeros(size(w));
%calculate cost
for i = 1:n
    L = L + X(:,i)'* w(:,y(i) + 1);
    
end

L = L - sum(log(sum(exp(X'* w),2)));
exponent = exp(X'*w); %5000 x 10

for t = 1:classNum
    for i = 1:n
        grad(:,t) = grad(:,t) + X(:,i) * ( (y(:,i) + 1 == t) - (exponent(i,t)/sum(exponent(i,:))));
    end
end

end