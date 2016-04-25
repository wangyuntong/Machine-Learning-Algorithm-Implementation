% Implementation of NonNegative matrix factorization using Euclidean
% Penalty
clc;
clear;
load faces.mat

% Seld-defined parameters
rank = 25;
iteration = 200;
small = 1e-16;

% Initialize W H ~ uniform(0,1)
[dim, object] = size(X);
W = rand(dim, rank); % 1024x25
H = rand(rank, object); % 25x1000
sqerr = zeros(1, iteration);
% Update H adn W
for iter = 1 : iteration
    % add a small number to denominator to avoid dividing zero.
    WTX = W.' * X; 
    WTW = W.' * W; % 25x25
    WTWH = W.' * W * H + small;
    % Update H
    for k = 1 : rank
        for j = 1 : object
            % WTWH = WTW(k, :) * H(:, j) + small;
            % H(k, j) = H(k, j) * WTX(k, j) / WTWH;
            H(k, j) = H(k, j) * WTX(k, j) / WTWH(k, j);
        end
    end
    
    XHT = X * H.';
    HHT = H * H.';
    WHHT = W * H * H.' + small;
    % Update W
    for i = 1 : dim
        for k = 1 : rank
            % WHHT = W(i, :) * HHT(:, k) + small;
            % W(i, k) = W(i, k) * XHT(i, k) / WHHT;
            W(i, k) = W(i, k) * XHT(i, k) / WHHT(i, k);
        end
    end
    
    % Record the object function
    sqerr(iter) = sum(sum((X - W*H).^2));
end

figure
plot(sqerr);
title('Evolution of objective function (square error) in terms of iteration');

% pick 10 columns from W 1~10
% Plot the image in W and the corresponding column in X
figure
p = 1;
for i = [0, 2]
    for j = 1 : 5
        subplot(4, 5, i*5 + j)
        imshow(reshape(W(:,p),32, 32));
        p = p + 1;
    end
end

[maxWeight, Idx] = max(H, [], 2);
selectedX = X(:, Idx(1:10));

p = 1;
for i = [1, 3]
    for j = 1 : 5
        subplot(4, 5, i*5 + j)
        imshow(reshape(selectedX(:,p)/255,32, 32));
        p = p + 1;
    end
end
title('Image with highest weight in the result (row 1 and row 3) and their corresponding columns of X (row 2 and row 4)');
