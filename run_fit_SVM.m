if ispc
    addpath(genpath('P:/code/development/utilities/MachineLearning'));
else
addpath(genpath('/group_shares/PSYCH/code/development/utilities/MachineLearning'));
end
%% Make fake data
n_cases=20;% define the number of participant, of cases
n_feat=200;% define the number of features


X1=randn(n_cases/2,n_feat); % Fake features for half cases
X2=10+randn(n_cases/2,n_feat); % Fake distinct features for second half of cases
X=[X1; X2];


y=(1:n_cases)'>n_cases/2;% Assign classes (This will make a vector of 1's and 0's)

%% Define options for the code
options=[];



options.N=7;
options.core_features=8;
options.increment_features=80;
options.upto_features=8;

[DATA, NULL, options]=basic_SVM(X,y,options);
