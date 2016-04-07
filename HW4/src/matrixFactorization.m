function [U, V, RMSE,logLikelihood] = matrixFactorization(sigma_square, dim, lambda, object, user, testData, iteration)
%% Matrix Factorization Function
% Input:
% sigmaSquare : 0.25, sigma^2 paramaeter of M's distribution
% dim: dimension of features
% lambda: 10, lambda parameter of U, V distribution
% object: struct, for each cell: users who have rated this object, user ratings for this object
% user: struct, for each cell: movies that have been rated for this user, ratings of movies for this user
% testData: 5000 x 3 , each row: user_id, movie_id, corresponding ratings,
% total 5000 test ratings
% iteration: iteration number

% Output:
% U : dim x N_user , user location represents user preference on each
% feature
% V :  dim x N_object, object location represents feature weight of each
% object
% RMSE : iteration x 1, root mean square error for each iteration
% logLikelihood : iteration x 1,  log likelihood for each iteration



    N_object = size(object,2);
    N_user = size(user, 2);
    I_dim = eye(dim);
    RMSE = zeros(iteration,1);
    logLikelihood = zeros(iteration,1);

    % Initialize V with multivariate normal N(0, lambda*I)
    V = mvnrnd(zeros(1, dim), lambda * I_dim, N_object).';
    U = zeros(dim, N_user);

    % Start iteration
    for iter = 1 : iteration
        % Part 1: Update U
        for i = 1 : N_user
            V_user = V(:, user(i).movie_id); % dim x V_user_number
            % d x 1 = d x d * d x 1
            U(:,i) = (lambda * sigma_square * I_dim + V_user * V_user.') \ sum( repmat(user(i).rating, dim, 1) .* V_user, 2);
        end
        % Part 2: Update V
        for j = 1 : N_object
            U_object = U(:, object(j).user_id); % dim x U_object_number
            V(:,j) = (lambda * sigma_square * I_dim + U_object * U_object.') \ sum( repmat(object(j).rating, dim, 1) .* U_object, 2);
        end

        % prediction testing error
        pred = diag(U(:,testData(:,1)).' * V(:,testData(:,2))); % 5000 x 1
        pred = round(pred);
        for i = 1 : size(testData,1)
            if (pred(i,1) > 5)
                pred(i,1) = 5;
            elseif (pred(i,1) < 1)
                pred(i,1) = 1;
            end
        end
        % Calculate RMSE
        RMSE(iter,1) = sqrt(mean((pred - testData(:,3)).^2));
        % Calculate log likelihood
        for i = 1 : N_object
            logLikelihood(iter,1) = logLikelihood(iter,1) - 1 / (2*sigma_square) * sum((object(i).rating - V(:,i).' * U(:,object(i).user_id)).^2);
        end
        for i = 1 : N_object
            logLikelihood(iter,1) = logLikelihood(iter,1) - lambda / 2 * sum(V(:,i).^2);
        end
        for i = 1 : N_user
            logLikelihood(iter,1) = logLikelihood(iter,1) - lambda / 2 * sum(U(:,i).^2);
        end       
    end
end


        