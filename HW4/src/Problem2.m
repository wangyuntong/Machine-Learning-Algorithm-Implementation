clc;clear;

load movie_ratings.mat;

% Input
sigmaSquare = 0.25;
dim = 10;
lambda = 10;
iteration = 100;

% Perform Matrix Factorization
[U, V, RMSE,logLikelihood] = matrixFactorization(sigmaSquare, dim, lambda, movie, user, ratings_test, iteration);

% Plot RMSE on iteration
figure
plot(RMSE);
title('RMSE of Prediction on Iteration');

% Plot log likelihood on iteration
figure; 
plot(logLikelihood);
title('Log Likelihood of Prediction on Iteration');

% Find 5 nearest neighbors of 3 well-known movies
% 56. {'Pulp Fiction (1994)'}
% 69. {'Forrest Gump (1994)'}
% 313. {'Titanic (1997)'}

famous_movie_index = [56, 69, 313];
for i = 1 : size(famous_movie_index,2)
    movie1 = V(:,famous_movie_index(i));
    dist = sqrt(sum((V - repmat(movie1,1,size(V,2))).^2,1));
    [nearDistances,index] = getNSmallestElements(dist,6);
    name_results = movie_names(index);
    fprintf('Query movie: %s\n',name_results{1});
    near_size = 5;
    for j = 1 : near_size
        fprintf('No. %d closest movie: %s, distance: %f\n',j, name_results{j + 1}, nearDistances(1,j + 1));
    end
    fprintf('\n');   
end

% Perform K-means on User location for 100 times and pick the one with smallest cost
K = 20;
kmeans_iter = 50;
iter_avoid_local_min = 100;
N_object = size(movie,2);
N_user = size(user, 2);
    
Mu_res_user = zeros(K,dim);
c_res_user = zeros(1,N_user);
kmeans_cost_user_res = 100000 * ones(1,kmeans_iter);
for i = 1 : iter_avoid_local_min
    [Mu, c, kmeans_cost_user] = kmeans(K, U.', kmeans_iter);
    if(kmeans_cost_user(1,kmeans_iter) < kmeans_cost_user_res(1,kmeans_iter))
        Mu_res_user = Mu;
        c_res_user = c;
        kmeans_cost_user_res = kmeans_cost_user;
    end
end
figure
plot(kmeans_cost_user_res);
title('Kmeans cost of user location with K = 20');
figure;
histogram(c_res_user);
title('Kmeans results of user location on each cluster');
% Find 5 centroids with most frequency
centroid_num = 5;
[userCentroidsFreq, userCentroidsIdx] = getNLargestElements(histc(c_res_user,1:K),centroid_num);
for i = 1 : centroid_num
    fprintf('No. %d popular centroid: ',i);
    disp(Mu_res_user(userCentroidsIdx(i),:));
    fprintf('The number of users allocated to the centroid: %d\n',userCentroidsFreq(i));
    % Find 10 movies most related to the centroid
    popular_size = 10;
    dot_prod = Mu_res_user(userCentroidsIdx(i),:) * V;
    [popularity, popularIdx] = getNLargestElements(dot_prod, popular_size);
    for j = 1 : popular_size
        name_cell = movie_names(popularIdx(j));
        fprintf('No. %d popular movie of this centroid: %s, distance: %f\n',j, name_cell{1}, popularity(j));
    end   
    fprintf('\n');        
end

% Perform K-means on Movie location for 100 times and pick the one with smallest cost
Mu_res_movie = zeros(K,dim);
c_res_movie = zeros(1,N_object);
kmeans_cost_movie_res = 100000 * ones(1,kmeans_iter);
for i = 1 : iter_avoid_local_min
    [Mu, c, kmeans_cost_movie] = kmeans(K, U.', kmeans_iter);
    if(kmeans_cost_movie(1,kmeans_iter) < kmeans_cost_movie_res(1,kmeans_iter))
        Mu_res_movie = Mu;
        c_res_movie = c;
        kmeans_cost_movie_res = kmeans_cost_movie;
    end
end
figure
plot(kmeans_cost_movie_res);
title('Kmeans cost of movie location with K = 20');
figure;
histogram(c_res_movie);
title('Kmeans results of movies location on each cluster');

% Find 5 centroids with most frequency
centroid_num = 5;
[movieCentroidsFreq, movieCentroidsIdx] = getNLargestElements(histc(c_res_movie,1:K),centroid_num);
for i = 1 : centroid_num
    fprintf('No. %d popular centroid: ',i);
    disp(Mu_res_movie(movieCentroidsIdx(i),:));
    fprintf('The number of movies allocated to the centroid: %d\n',movieCentroidsFreq(i));
    % Find 10 movies most related to the centroid
    movie1 = V(:,movieCentroidsIdx(i));
    dist = sqrt(sum((V - repmat(movie1,1,size(V,2))).^2,1));
    near_size = 10;
    [nearDistances,index] = getNSmallestElements(dist,near_size + 1);
    name_results = movie_names(index);
    for j = 2 : near_size
        fprintf('No. %d closest movie: %s, distance: %f\n',j-1, name_results{j}, nearDistances(1,j));
    end
    fprintf('\n');        
end