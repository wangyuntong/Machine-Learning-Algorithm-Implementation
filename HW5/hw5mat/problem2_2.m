% Implementation of NonNegative matrix factorization using Divergence
% Penalty
clc;
clear;
load nyt_data.mat

% Seld-defined parameters
rank = 25;
iteration = 200;
small = 1e-16;

% Initialize W H ~ uniform(0,1)
object = size(Xcnt, 2);
dim = size(nyt_vocab, 1);
X = zeros(dim, object);

% Create matrix X, Xij means word i appears Xij times in document j
for d = 1 : object
    for i = 1 : size(Xid{d}, 2)
        X(Xid{d}(i), d) = Xcnt{d}(i);
    end
end

W = rand(dim, rank); % 3012x25
H = rand(rank, object); % 25x8447
div = zeros(1, iteration);

% time = 0;
% Update H adn W
for iter = 1 : iteration
    % add a small number to denominator to avoid dividing zero.
    % Update H
    % t0=clock;
    nominator = W' * (X ./ ((W * H) + small) );
    denominator = repmat(sum(W, 1).' , 1, object) + small;
    H = H .* nominator ./ denominator;
    
    % Update W
    nominator = (X ./ ((W * H) + small) ) * H.'; % dim x object * object x rank = dim x rank 
    denominator = repmat(sum(H, 2).' , dim, 1) + small;
    W = W .* nominator ./ denominator;
    
    % Record the object function
    WH = W * H;
    div(iter) = sum(sum(X.*log(1./(WH + small)) + WH));
    % time = time + etime(clock,t0);
    % fprintf('Iteration %d, time elapsed : %.4f\n', iter, time);
end

figure
plot(div);
title('Evolution of objective function (divergence) in terms of iteration');

% Normalize W, s.t. each column sum to 1
W = W./repmat(sum(W, 1), dim, 1);

% pick 10 columns from W 1~10
select = 1:10;
[sortWeight, weightIdx] = sort(W(:, select), 1, 'descend');

for i = 1 : 10
	fprintf('For topic (column) %d, the most dominant words are: \n', i);
    fprintf('          Word       Probability\n');
	for j = 1 : 10
		fprintf ('No %d.     %s      %.4f \n', j, nyt_vocab{weightIdx(j, i)}, sortWeight(j, i));
    end
    fprintf('\n');
end

