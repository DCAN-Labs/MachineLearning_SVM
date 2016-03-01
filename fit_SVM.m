function [SVMModel, cv, ix] = fit_SVM(X,y,partition, N_opt_svm)


[n, feat]=size(X);
if isnumeric(partition)
    cv = cvpartition(y,'k',20);
else
    cv = cvpartition(y,'LeaveOut');
end




% p=1-fraction_in_sample;
% cv = cvpartition(y,'HoldOut',p);
% if n<20
%     cv = cvpartition(n,'LeaveOut');
% else
%     cv = cvpartition(y,'k',20);
% end
cdata=X;
grp=y;

%% for partition stuff
% ix=feat_selection(cdata',grp,method);

ix=1:feat;
%% make the optimization for sigmoid kernel
minfn = @(z)kfoldLoss(fitcsvm(cdata(:,ix(1:feat)),grp,'CVPartition',cv,...
    'KernelFunction','rbf','BoxConstraint',exp(z(2)),...
    'KernelScale',exp(z(1))));

m = N_opt_svm;

seed=4*rand(2,m);
if m<4
    seed(2,1)=4;
else
    m_half=round(m/2);
    seed(2,m_half:-1:1)=linspace(0,4,m_half);% to make the first boxconstrain high (10^2) to penalize overfitting
end

% 



opts = optimset('TolX',5e-4,'TolFun',5e-4);
fval = zeros(m,1);
z = nan(m,2);
% tic
for j = 1:m;
    try 
        [searchmin, fval(j)] = patternsearch(minfn,seed(:,j),opts);
    catch
        [searchmin, fval(j)] = fminsearch(minfn,seed(:,j),opts);
    end   
    z(j,:) = exp(searchmin);
%     [j fval(j) toc/60];
    if fval(j)==0
        break
    end
end

z = z(find(fval == min(fval),1),:);
SVM_sig = fitcsvm(cdata(:,ix(1:feat)),grp,'KernelFunction','rbf',...
    'KernelScale',z(1),'BoxConstraint',z(2));
% loocv_e=kfoldLoss(crossval(SVM_sig));
SVMModel = SVM_sig;
%% calculate accuracy


