function [acc_in, acc_out, sens_in, sens_out, spec_in, spec_out, feat] = get_predictions(i, perc_training, n_cases, y, X, feat_sets,classes,partition, N_opt_svm, balance_1_0,is_null)

if is_null % randomize the classes
    y(randperm(n_cases))=y;
end

if balance_1_0
    
    nc1=sum(y==classes(1));
    nc2=sum(y==classes(2));
    n_min=min(nc1,nc2);
    ix_c1=find(y==classes(1));
    ix_c2=find(y==classes(2));
    
    ix_c1=ix_c1(randperm(nc1));
    ix_c2=ix_c2(randperm(nc2));
    
    y_trunc_c1=y(ix_c1(1:n_min));
    y_trunc_c2=y(ix_c2(1:n_min));
    y_trunc=[y_trunc_c1;y_trunc_c2];
    
    y_extra_c1=y(ix_c1(n_min+1:end));
    y_extra_c2=y(ix_c2(n_min+1:end));
    y_extra=[y_extra_c1;y_extra_c2];
    
    
    X_trunc_c1=X(ix_c1(1:n_min),:);
    X_trunc_c2=X(ix_c2(1:n_min),:);
    X_trunc=[X_trunc_c1;X_trunc_c2];
    
    X_extra_c1=X(ix_c1(n_min+1:end),:);
    X_extra_c2=X(ix_c2(n_min+1:end),:);
    X_extra=[X_extra_c1;X_extra_c2];
    
else
    y_trunc=y;
    X_trunc=X;
    y_extra=[];
    X_extra=[];
end


% if is_null % randomize the classes
%     y_trunc(randperm(length(y_trunc)))=y_trunc;
%     y_extra(randperm(length(y_extra)))=y_extra;
% end

% ix=randperm(n_cases);
% ix_in=ix(1:n_in);
% ix_out=ix(n_in+1:end);

cv=cvpartition(y_trunc,'HoldOut',1-perc_training/100);

ix_in=find(cv.training);
ix_out=find(cv.test);

y_in=y_trunc(ix_in);
y_out=y_trunc(ix_out);
X_in=X_trunc(ix_in,:);
X_out=X_trunc(ix_out,:);

%
X_KS=[X_in;X_extra];
y_KS=[y_in;y_extra];

X_class1=X_KS(y_KS==classes(1),:);
X_class2=X_KS(y_KS==classes(2),:);

sort_feat_kolmogorov = sort_feat_kolmogorov_difference(X_class1,X_class2,0);
ix_feat=sort_feat_kolmogorov(1:feat_sets(i));
feat=zeros(1,size(X,2));

feat(ix_feat)=1;

X_train=X_in(:,ix_feat);
X_test=X_out(:,ix_feat);
[SVMModel] = fit_SVM(X_train,y_in,partition, N_opt_svm);

pin = predict(SVMModel, X_train);
pout = predict(SVMModel, X_test);


acc_in=mean(double(pin == y_in));
acc_out=mean(double(pout == y_out));


sens_in=mean(double(pin(y_in==1)));
sens_out=mean(double(pout(y_out==1)));

spec_in=1-mean(double(pin(y_in==0)));
spec_out=1-mean(double(pout(y_out==0)));
