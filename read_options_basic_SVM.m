function [N, perc_training, core_features, increment_features, upto_features,partition, N_opt_svm, null_hypothesis,balance_1_0,options] = read_options_basic_SVM(options,n_cases, n_feat)
%% Oscar Miranda-Dominguez
% Jan 7, 2016
% N, how many times the analysis will be repeated for each feature set
%
% perc_training, percentage of the sample to be used for training
%
% core_features, how many features to include in the first feature set
%
% increment_features, how many features to include in the subsequent
% feature sets
%
% upto_features, until how many features to include
%
% partition, which partition method to use for each in-sample training:
% loocv default)10 fold, 20-fold, ...

%% Assign default values, if not provided and error check

% Number of repetitions for each experiment
if ~isfield(options,'N') || isempty(options.N);
    options.N=1000;
elseif options.N<1
    options.N=1000;
end;
options.N=round(options.N);

% Pecentile, 0 to 100 to be included in training
if ~isfield(options,'perc_training') || isempty(options.perc_training);
    options.perc_training=80;
elseif options.perc_training<0
    options.perc_training=10;
elseif options.perc_training>=100
    options.perc_training=99;
end;
options.perc_training=options.perc_training;

% Define the minimum set of core_features to start doing SVM
if ~isfield(options,'core_features') || isempty(options.core_features);
    options.core_features=round(n_feat/10);
elseif options.core_features>n_feat
    options.core_features=n_feat;
elseif options.core_features<0
    options.core_features=round(n_feat/10);
end;
options.core_features=round(options.core_features);

% Define how many features to add per set
if ~isfield(options,'increment_features') || isempty(options.increment_features);
    options.increment_features=round(n_feat/10);
end;
options.increment_features=round(options.increment_features);

% Define until how many features to include
if ~isfield(options,'upto_features') || isempty(options.upto_features);
    options.upto_features=n_feat;
elseif options.upto_features > n_feat
    options.upto_features=n_feat;
elseif options.upto_features<0;
    options.upto_features=n_feat;
end;
options.upto_features=round(options.upto_features);

% Define the minimum set of core_features to start doing SVM
if ~isfield(options,'partition') || isempty(options.partition);
    options.partition='LeaveOut';
elseif options.partition>n_cases
    options.partition='LeaveOut';
elseif options.partition<0
    options.partition='LeaveOut';
end;


% Define how many to optimize svm
if ~isfield(options,'N_opt_svm') || isempty(options.N_opt_svm);
    options.N_opt_svm=1;
elseif options.N_opt_svm<0;
    options.N_opt_svm=1;
end;
options.N_opt_svm=round(options.N_opt_svm);


% Define if Null hypothesis will be calculated, if not provided, by default
% will use the number of features with the best out of sample performance
% to calculate the null hypothesis
if ~isfield(options,'null_hypothesis') || isempty(options.null_hypothesis);
    options.null_hypothesis='best';
elseif ~and(strcmp(options.null_hypothesis,'all'),strcmp(options.null_hypothesis,'none'))
    options.null_hypothesis='best';
end

% Define if ones and zeros need to be balanced
if ~isfield(options,'balance_1_0') || isempty(options.balance_1_0);
    options.balance_1_0=1;
elseif options.balance_1_0>1
    options.balance_1_0=1;
elseif options.balance_1_0<0
    options.balance_1_0=0;
end;
options.balance_1_0=round(options.balance_1_0);
%% unfold variables
N=options.N;
perc_training= options.perc_training;
core_features=options.core_features;
increment_features=options.increment_features;
upto_features=options.upto_features;
partition=options.partition;
N_opt_svm=options.N_opt_svm;
balance_1_0=options.balance_1_0;
null_hypothesis=options.null_hypothesis;
