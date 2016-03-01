function [DATA, NULL, options]=basic_SVM(X,y,options)
%% Oscar Miranda Dominguez
% Jan 2016
%% This function performs SVM N times on the feature data set X, with the binary classes defined in y
% X's size is expected to be cases x features (a matrix)
% y's size is expected to be cases x 1 ( a binary vector)

% TO use defaults, use just 2 input arguments or define options=[];
%
rng('shuffle') 
%% Count features and cases & error check
[n_cases, n_feat]=size(X);
[s1, s2]=size(y);
if s1~=n_cases||s2~=1
    error('Double check X & y size, X''s size is expected to be cases x features (a matrix), y''s size is expected to be cases x 1 ( a binary vector)')
end

classes=unique(y,'rows');
n_classes=length(classes);
if n_classes~=2
    error(['Double check the output, it needs to have 2 classes, vector provided has' num2str(n_classes) ' class(es)'])
end

display(['This data set has ' num2str(n_cases) ' cases and ' num2str(n_feat) ' features'])
1;

if nargin < 3
    options=[];
end
%% Read options

[N, perc_training, core_features, increment_features, upto_features,partition, N_opt_svm, null_hypothesis,balance_1_0,options] = read_options_basic_SVM(options,n_cases, n_feat);
options
%% Count feature sets

feat_sets=core_features:increment_features:upto_features;
if feat_sets(end)~=upto_features
    feat_sets(end+1)=upto_features;
end
n_feat_sets=numel(feat_sets);
%% Memory pre-allocation

acc_in=zeros(n_feat_sets,N);
acc_out=zeros(n_feat_sets,N);

sens_in=zeros(n_feat_sets,N);
sens_out=zeros(n_feat_sets,N);

spec_in=zeros(n_feat_sets,N);
spec_out=zeros(n_feat_sets,N);

feat=nan(n_feat,n_feat_sets,N);
%% identify n featature sets for null hypothesis

switch options.null_hypothesis
    case 'best'
        n_null_feat_sets=1;
    case 'all'
        n_null_feat_sets=n_feat_sets;
    case 'none'
        n_null_feat_sets=0;
end
%% Memory pre-allocation for Null

if n_null_feat_sets>0
    feat_null=nan(n_feat,n_null_feat_sets,N);
end

acc_in_null=zeros(n_null_feat_sets,N);
acc_out_null=zeros(n_null_feat_sets,N);

sens_in_null=zeros(n_null_feat_sets,N);
sens_out_null=zeros(n_null_feat_sets,N);

spec_in_null=zeros(n_null_feat_sets,N);
spec_out_null=zeros(n_null_feat_sets,N);

%% DO work
is_null=0;
for i=1:n_feat_sets
    display(['Running set ' num2str(i) ' out of ' num2str(n_feat_sets)])
    mA=0;
    for j=1:N
        tic
        [acc_in(i,j) acc_out(i,j) sens_in(i,j) sens_out(i,j) spec_in(i,j) spec_out(i,j) feat(:,i,j)] = get_predictions(i, perc_training, n_cases, y, X, feat_sets,classes,partition, N_opt_svm,balance_1_0,is_null);
        foo=toc;
        mA=acc_out(i,j)/j+mA*(j-1)/j;
        display(['    Run ' num2str(j) ' took ' num2str(foo) ' seconds (Accuracy = ' num2str(acc_out(i,j),'%4.4f') ', cumulative accuracy = ' num2str(mA,'%4.4f') ')'])
    end
    
end
display('Completed!')

% mean(acc_out,2)
% mean(acc_in,2)

% Save variables for ploting
y_in(:,1)=mean(acc_in,2);
y_in(:,2)=mean(sens_in,2);
y_in(:,3)=mean(spec_in,2);

y_out(:,1)=mean(acc_out,2);
y_out(:,2)=mean(sens_out,2);
y_out(:,3)=mean(spec_out,2);

% Pack data
DATA.acc_in=acc_in;
DATA.sens_in=sens_in;
DATA.spec_in=spec_in;
DATA.acc_out=acc_out;
DATA.sens_out=sens_out;
DATA.spec_out=spec_out;
DATA.feat=feat;
%% Do null hypothesis
if strcmp(options.null_hypothesis,'none')
    null_in(1)=nan;
    null_in(2)=nan;
    null_in(3)=nan;
    
    null_out(1)=nan;
    null_out(2)=nan;
    null_out(3)=nan;
else
    
    [B, IX]=max(mean(acc_out,2)); % Identify the best out_of_sample performance
    if n_null_feat_sets==1
        feat_sets_null=feat_sets(IX);
        temp_ix=1;
    end
    
    if n_null_feat_sets>1
        feat_sets_null=feat_sets;
        temp_ix=IX;
    end
    
    
    is_null=1;
    for i=1:n_null_feat_sets
        display(['Running NULL set ' num2str(i) ' out of ' num2str(n_null_feat_sets)])
        mA=0;
        for j=1:N
            tic
            [acc_in_null(i,j) acc_out_null(i,j) sens_in_null(i,j) sens_out_null(i,j) spec_in_null(i,j) spec_out_null(i,j) feat_null(:,i,j)] = get_predictions(i, perc_training, n_cases, y, X, feat_sets_null,classes,partition,N_opt_svm,balance_1_0,is_null);
            foo=toc;
            mA=acc_out_null(i,j)/j+mA*(j-1)/j;
            display(['    Run ' num2str(j) ' took ' num2str(foo) ' seconds (Accuracy = ' num2str(acc_out_null(i,j),'%4.4f') ', cumulative accuracy = ' num2str(mA,'%4.4f') ')'])
        end
    end
    display('Completed!')
    
    null_in(1)=mean(acc_in_null(temp_ix,:));
    null_in(2)=mean(sens_in_null(temp_ix,:));
    null_in(3)=mean(spec_in_null(temp_ix,:));
    
    null_out(1)=mean(acc_out_null(temp_ix,:));
    null_out(2)=mean(sens_out_null(temp_ix,:));
    null_out(3)=mean(spec_out_null(temp_ix,:));
end

% Pack data
NULL.acc_in_null=acc_in_null;
NULL.sens_in_null=sens_in_null;
NULL.spec_in_null=spec_in_null;
NULL.acc_out_null=acc_out_null;
NULL.sens_out_null=sens_out_null;
NULL.spec_out_null=spec_out_null;
NULL.feat_null=feat_null;
%% plotting

f = figure('Visible','on',...
    'Units','centimeters',...
    'PaperUnits','centimeters',...
    'name','Classification results',...
    'PaperPosition',[8 1 16 8],...
    'Position',[8 1 16 8]);


fs_axis=8; %size of fonts in plots
fs_title=12;%size of fonts in title
fs_legend=10;
fs_label=9;

l_style='--';

set(gcf,'Color','w',...
    'DefaultAxesLineWidth',1,...
    'DefaultAxesFontSize',fs_axis)%,...

my_color=winter(n_feat_sets);

marker_type{1}='+';% for acc
marker_type{2}='.'; %for sp
marker_type{3}='s';% for sn

% Define marker size
ms(1)=4;% for acc
ms(2)=16;%for sp
ms(3)=4;% for sn

%%
subplot 221
% IX=3
% color_highlight=[255,237,160]/255;
%
% color_highlight=cool(n_feat_sets);
% color_highlight=color_highlight(IX,:);
color_highlight=[0.6667    0.3333    1.0000];
i=0;
plot(i,null_in(1),marker_type{1},'markersize',ms(1),'color','r')
patch([IX-0.1 IX+.1 IX+.1 IX-.1],[0.01 0.01 0.99 0.99],color_highlight,...
    'EdgeColor',color_highlight)
hold all
plot(i,null_in(2),marker_type{2},'markersize',ms(2),'color','r')
plot(i,null_in(3),marker_type{3},'markersize',ms(3),'color','r')
for i=1:n_feat_sets
    plot(i,y_in(i,1),marker_type{1},'markersize',ms(1),'color',my_color(i,:))
    plot(i,y_in(i,2),marker_type{2},'markersize',ms(2),'color',my_color(i,:))
    plot(i,y_in(i,3),marker_type{3},'markersize',ms(3),'color',my_color(i,:))
end
hold off
xlim([-.2 n_feat_sets+.2])
set(gca,'xtick',0:n_feat_sets)
set(gca,'xticklabel',[])
ylim([0 1])
set(gca,'ytick',0:.2:1)
set(gca,'yticklabel',num2str(get(gca,'ytick')','%4.1f'))
ylabel({'A1) In-sample ','Accuracy'},'fontsize',fs_label)
grid on
line([-1 n_feat_sets+1],[null_in([1 1])],'color','r','linestyle',l_style)
title(['A) Summary results, N = ' num2str(N)])
title(['A) Summary, ' num2str(n_cases) ' cases, ' num2str(N) ' rep.'])



subplot 223
i=0;
plot(i,null_out(1),marker_type{1},'markersize',ms(1),'color','r')
patch([IX-0.1 IX+.1 IX+.1 IX-.1],[0.01 0.01 0.99 0.99],color_highlight,...
    'EdgeColor',color_highlight)
hold all
plot(i,null_out(2),marker_type{2},'markersize',ms(2),'color','r')
plot(i,null_out(3),marker_type{3},'markersize',ms(3),'color','r')
x_leg=cell(n_feat_sets+1,1);
x_leg{1}='Null';
for i=1:n_feat_sets
    plot(i,y_out(i,1),marker_type{1},'markersize',ms(1),'color',my_color(i,:))
    plot(i,y_out(i,2),marker_type{2},'markersize',ms(2),'color',my_color(i,:))
    plot(i,y_out(i,3),marker_type{3},'markersize',ms(3),'color',my_color(i,:))
    x_leg{i+1}=num2str(feat_sets(i));
end
hold off
xlim([-.2 n_feat_sets+.2])
set(gca,'xtick',0:n_feat_sets)
set(gca,'xticklabel',[])
set(gca,'xticklabel',x_leg)
ax = gca;
ax.XTickLabelRotation = 90;
ylim([0 1])
set(gca,'ytick',0:.2:1)
set(gca,'yticklabel',num2str(get(gca,'ytick')','%4.1f'))
ylabel({'A2) Out-sample ','Accuracy'},'fontsize',fs_label)
grid on
line([-1 n_feat_sets+1],[null_out([1 1])],'color','r','linestyle',l_style)

ACC_IN=acc_in(IX,:);
ACC_OUT=acc_out(IX,:);

SENS_IN=sens_in(IX,:);
SENS_OUT=sens_out(IX,:);

SPEC_IN=spec_in(IX,:);
SPEC_OUT=spec_out(IX,:);

best=cat(3,[ACC_IN; ACC_OUT]',[SENS_IN; SENS_OUT]',[SPEC_IN; SPEC_OUT]');
null=cat(3,[acc_in_null(temp_ix,:); acc_out_null(temp_ix,:)]',[sens_in_null(temp_ix,:); sens_out_null(temp_ix,:)]',[spec_in_null(temp_ix,:); spec_out_null(temp_ix,:)]');

stit{1}='B) Full Acc.';
stit{2}='C) Sensit.';
stit{3}='D) Specif.';
lab=cell(4,1);
lab{2}='Null';
lab{3}=num2str(feat_sets(IX));
lab{1}=' ';
lab{4}=' ';

k=0;
my_color2=[1 0 0;my_color(IX,:)];

for j=1:2
    for i=1:3
        
        k=k+1;
        subplot(2,6,3+k);
        pos=get(gca,'position');
        
        plot([1 2],[mean(null(:,j,i)) mean(best(:,j,i))],'k.')
        line([1 1],prctile(null(:,j,i),[2.5 97.5]),'linewidth',1,'color',my_color2(1,:))
        line([1 1],prctile(null(:,j,i),[25 75]),'linewidth',3,'color',my_color2(1,:))
        
        line([2 2],prctile(best(:,j,i),[2.5 97.5]),'linewidth',1,'color',my_color2(2,:))
        line([2 2],prctile(best(:,j,i),[25 75]),'linewidth',3,'color',my_color2(2,:))
        hold all
        plot(1 ,mean(null(:,j,i)),'o','markeredgecolor',my_color2(1,:),'MarkerFaceColor','w','markersize',8)
        plot(1 ,mean(null(:,j,i)),marker_type{i},'markeredgecolor',my_color2(1,:),'MarkerFaceColor','w','markersize',4)
        %         plot(1 ,mean(null(:,j,i)),'.k')
        
        plot(2 ,mean(best(:,j,i)),'o','markeredgecolor',my_color2(2,:),'MarkerFaceColor','w','markersize',8)
        plot(2 ,mean(best(:,j,i)),marker_type{i},'markeredgecolor',my_color2(2,:),'MarkerFaceColor','w','markersize',4)
        %         plot(2 ,mean(best(:,j,i)),'.k')
        hold off
        
        set(gca,'position',pos)
        ylim([0 1])
        xlim([0 3])
        set(gca,'ytick',0:.2:1)
        set(gca,'yticklabel',num2str(get(gca,'ytick')','%4.1f'))
        
        grid on
        [h,p]=ttest2(best(:,j,i), null(:,j,i));
        
        text(mean(xlim),0.1,['p = ' num2str(p,'%4.2e')],...
            'HorizontalAlignment','center',...
            'fontsize',fs_axis)
        
        if i<3
            set(gca,'yticklabel',[])
            %             title(tit{j})
        end
        if j<2
            set(gca,'XTickLabel',{' '})
            
        else
            set(gca,'XTickLabel',lab)
            ax = gca;
            ax.XTickLabelRotation = 90;
        end
        if i==3
            set(gca,'yaxislocation','right')
        end
        
        if j==1
            title(stit{i},'fontsize',fs_legend)
        end
        
    end
    k=k+3;
    
end


%%
% Randomize partition
%         ix=randperm(n_cases);
%         ix_in=ix(1:n_in);
%         ix_out=ix(n_in+1:end);
%
%         y_in=y(ix_in);
%         y_out=y(ix_out);
%         X_in=X(ix_in,:);
%         X_out=X(ix_out,:);
%
%         %
%         X_class1=X_in(y_in==classes(1),:);
%         X_class2=X_in(y_in==classes(2),:);
%
%         sort_feat_kolmogorov = sort_feat_kolmogorov_difference(X_class1,X_class2,0);
%         ix_feat=sort_feat_kolmogorov(1:feat_sets(i));
%         feat(1:feat_sets(i),i,j)=ix_feat;
%
%         X_train=X_in(:,ix_feat);
%         X_test=X_out(:,ix_feat);
%         [SVMModel] = fit_SVM(X_train,y_in,partition, N_opt_svm);
%
%         pin = predict(SVMModel, X_train);
%         pout = predict(SVMModel, X_test);
%
%
%         acc_in(i,j)=mean(double(pin == y_in));
%         acc_out(i,j)=mean(double(pout == y_out));
%
%
%         sens_in(i,j)=mean(double(pin(y_in==1)));
%         sens_out(i,j)=mean(double(pout(y_out==1)));
%
%         spec_in(i,j)=1-mean(double(pin(y_in==0)));
%         spec_out(i,j)=1-mean(double(pout(y_out==0)));
