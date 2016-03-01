%% Basic settings
N=1000; % Number of repetitions for each experiment
perc_trainin=80;   %Pecentile, 0 to 100 to be included in training
n_features=10;     % How many features to include, from 1 until the maximum number of features
increment_features % How many features to add to the initial n_features to determine the optimal number of features
%% Advanced settings
% Training on each individual run
% Use LeaveOut for loocv, or a numerical value k for k-fold validation 
partition='LeaveOut'; % This will use loocv
% partition='10'; % This will use leave 10 cross validation
% partition='20'; % This will use leave 20 cross validation
