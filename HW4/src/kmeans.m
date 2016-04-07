function [Mu, c, cost] = kmeans(K, data, iteration)
%% K-means function

% input:    clustering number K
%           data, data_size * dim
%           iteration number

% output:   Mu: Centroid Mu, K*dim
%           c: Assigned cluster label, 1*data_size
%           cost: objective function, 1*iteration

    data_size = size(data,1);
    dim = size(data,2);
    c = zeros(1,data_size);
    cost = zeros(1,iteration);
    % Randomly Initialize Mu from dataset
    Mu = data(250 + round(250*rand(1,K)),:);
    
    for i = 1 : iteration
        % Part 1: Update c
        distance = zeros(data_size,K);
        for k = 1 : K
            distance(:,k) = sum((data - repmat(Mu(k,:),data_size, 1)).^2, 2); % data_size * 1
        end
        
        % Find the smallest distance for each data point
        [min_dist,new_c] = min(distance,[],2);
        
        % Check if results have converged
%         if (new_c.' == c)
%             break;
%         else
            c = new_c.';
%         end

        % Part 2: Update Mu
        for k = 1 : K
            k_cluster = data(c == k,:);
            Mu(k,:) = mean(k_cluster,1);
        end
        
        % Update objective function
        for k = 1 : K
            k_cluster = data(c == k,:);
            cost(1,i) = cost(1,i) + sum(sum((k_cluster - repmat(Mu(k,:),size(k_cluster,1), 1)).^2,1),2);
        end
        
    end  
end