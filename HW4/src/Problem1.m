clc;clear;

data_size = 500;
% Build a mixture gaussion ditribution
mu = [0,0;3,0;0,3];
Sigma = [1,1];
% Weight on each gaussion
p = [0.2,0.5,0.3];
obj = gmdistribution(mu, Sigma, p);

data = random(obj, data_size);

% Plot the data
figure
title('Gaussian Mixture Data Visualization');
scatter(data(:,1),data(:,2));

% Iterate for 20 times, data may converge before that
iter = 20;
cost = zeros(4,iter); % cost for 4 k value regarding iteration number
for K = 2:5
    [Mu, c, cost(K-1,:)] =kmeans(K, data, iter);
    % Mu : k*d, centroid
    % c : 1*data_size, cluster label
    % data : data_size*d
    if K == 3 || K == 5
        figure
        scatter(data(:,1),data(:,2),[],c,'filled');
        title(['K-means Cluster Results when K = ' num2str(K)]);
    end
end
x = 1:iter;
figure
plot(x,cost(1,:),x,cost(2,:),x,cost(3,:),x,cost(4,:));
legend('K=2','K=3','K=4','K=5');
title('K-means objective function revolution on iteration');
