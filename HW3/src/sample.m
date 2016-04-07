function c = sample(n, w)
% n : positive interger
% w : 1 x k vector, k-dimensional probability distribution
% return c : 1 x n vector, c(i) is in {1,2,...,k}, with Prob(c(i) = j|w) = w(j)
cdf = cumsum(w);
c = zeros(1,n);
for i = 1 : n
    c(1,i) = sum(rand > [0 , cdf(1:end-1)]);
end
end