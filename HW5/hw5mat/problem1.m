clc;clear;
load CFB2015.mat

%Initialize transition mastrix M
teamNum = 759;
M = zeros(teamNum);
for i = 1 : size(scores,1)
    team1Ind = scores(i,1);
    team1Score = scores(i,2);
    team2Ind = scores(i,3);
    team2Score = scores(i,4);
    % Update
    totalScore = team1Score + team2Score;
    M(team1Ind, team1Ind) = M(team1Ind, team1Ind) + (team1Score > team2Score) + team1Score / totalScore;
    M(team2Ind, team2Ind) = M(team2Ind, team2Ind) + (team2Score > team1Score) + team2Score / totalScore;  
    M(team2Ind, team1Ind) = M(team2Ind, team1Ind) + (team1Score > team2Score) + team1Score / totalScore;
    M(team1Ind, team2Ind) = M(team1Ind, team2Ind) + (team2Score > team1Score) + team2Score / totalScore;
    
end

% normalize so each row of M sum to 1
M = spdiags (sum (abs(M),2), 0, teamNum, teamNum) \ M;

% Initialize w0 with uniform distribution
w0 = ones(1, teamNum) / teamNum;

% Iterate T steps
T = 2500;
res = zeros(T, teamNum);
res(1,:) = w0;
for t = 2:T
    wt = res(t - 1, : ) * M;
    res(t,:) = wt / sum(wt);
end

[res, Idx] = sort(res,2,'descend');
top = 25;
disp = [10,100,1000,2500];
for i = 1 : 4
    fprintf('Top 25 team ranks when t = %d\n' ,disp(i));
    for j = 1:top
        fprintf('No. %d , Team: %d %s , Score : %.4f\n',j, Idx(disp(i),j), char(legend(Idx(disp(i),j))),res(disp(i),j));
    end
    fprintf('\n');
end

% 
[V,D] = eig(M.');
[D,I] = sort(diag(D),'descend');
mu = V(:, I(1)).';
w = sort(mu / sum (mu),'descend');
    
for t = 1 : T
    error(t) = sum(abs(res(t,:) - w));
end
figure
plot(error);
title('Evolution of ||wt - w?|| ');
fprintf('The value of ||wt - w?|| %.4f.\n',error(T));
    