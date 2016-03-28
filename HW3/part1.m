% Part 1, plot
% 
w = [0.1, 0.2, 0.3, 0.4];

for n = [50,250,500]
    figure
    histogram(sample(n,w));
    title(['histogram of sampled vector ' num2str(n)]);
end
