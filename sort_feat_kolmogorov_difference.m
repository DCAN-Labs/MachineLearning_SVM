function [ix, d_sign, p_value] = sort_feat_kolmogorov_difference(Xs,Xo,show_figure)

%% Oscar Miranda-Dominguez
if nargin<3
    show_figure=0;
end
rois=size(Xs,2);
h=zeros(rois,1);
p_value=zeros(rois,1);
d=zeros(rois,1);
sib_gto=zeros(rois,1);% siblings greather than others

for ii=1:rois
    arg_sib=Xs(:,ii);
    arg_oth=Xo(:,ii);
    [h(ii), p_value(ii), d(ii)]=kstest2(arg_sib,arg_oth);
    median_sib=median(arg_sib);
    median_oth=median(arg_oth);
    sib_gto(ii)=median_sib>median_oth;
end

%% Assign sign
d_sign=d;
d_sign(sib_gto<1)=-d_sign(sib_gto<1);
[B, ix]=sort(d_sign,'descend');

%% audit distances



if show_figure
    fs_axis=16; %size of fonts in plots
    fs_title=12;%size of fonts in title
    fs_label=10;%size of fonts in title
    fs_legend=10;
    set(gcf,'Color',[0.97 0.97 0.97],...
        'DefaultAxesLineWidth',1,...
        'DefaultAxesFontSize',fs_axis)%,...
    subplot 212
    plotyy(1:rois,d_sign(ix),1:rois,log10(p_value(ix)))
    plot(1:rois,d_sign(ix),'k','linewidth',3)
    %     legend('Signed distance,','p value (log_1_0)')
    axis tight
    xlabel({'ROIs','(Features)'})
    ylabel('Distance')
    set(gca,'yticklabel',num2str(get(gca,'ytick')','%4.1f'))
    %
    ns=size(Xs,1);
    no=size(Xo,1);
    y_temp=repmat('Unrelated',ns+no,1);
    y_temp(1:ns,:)=repmat('Siblings ',ns,1);
    up_to=1;
    n=min([20 size(Xs,1) size(Xo,1)]);
    r_min=min([Xs(:);Xo(:)]);
    r_max=max([Xs(:);Xo(:)]);
    range=linspace(r_min,r_max,n);
    range=linspace(0,1,n);
    for ii=1:up_to
        subplot(4,up_to,ii)
        i=ix(ii);
        
        ys=histc(Xs(:,i),range);
        yo=histc(Xo(:,i),range);
        
        ys=ys/sum(ys);
        yo=yo/sum(yo);
        
        stairs(range,[ys],'color',[0 .447 .741],'linewidth',3)
        hold all
        stairs(range,[yo],'color',[.85 .325 .098],'linewidth',3)
        hold off
        
        
        %         boxplot([Xs(:,i); Xo(:,i)],y_temp)
        %         ylim([0 1])
        title(['ROI ' num2str(i)])
        ylabel({'Relative', 'abundance'})
        
        set(gca,'yticklabel',num2str(get(gca,'ytick')','%4.1f'))
        set(gca,'xticklabel',num2str(get(gca,'xtick')','%4.1f'))
        legend('Sibling','Unrelated',...
            'Location','NorthWest')
        legend boxoff
        
        subplot(4,up_to,ii+up_to)
        y1=cumsum(histc(Xs(:,i),range));
        y2=cumsum(histc(Xo(:,i),range));
        
        y1=y1/y1(end);
        y2=y2/y2(end);
        
        ix_local=find(abs(y1-y2)==max(abs(y1-y2)));
        
        plot([range(ix_local) range(ix_local)],[y1(ix_local) y2(ix_local)],...
            'color','k',...
            'linewidth',2)
        
        hold all
        %plot(get(gca, 'xlim'),[.5 .5],'k--')
        plot(range,y1,'color',[0 .447 .741],'linewidth',3)
        plot(range,y2,'color',[.85 .325 .098],'linewidth',3)
        
        
        hold off
        legend('Distance',...
            'Location','NorthWest')
        title('Cumulative distribution')
        ylabel({'Relative', 'abundance'})
        grid off
        set(gca,'xticklabel',num2str(get(gca,'xtick')','%4.1f'))
        set(gca,'yticklabel',num2str(get(gca,'ytick')','%4.1f'))
        legend boxoff
        
        %         i=ix(end+1-ii);
        %         boxplot([Xs(:,i); Xo(:,i)],y_temp)
        %         ylim([0 1])
        %         title(['ROI ' num2str(i)])
        %         grid
    end
    set(gcf,'color',[1 1 1]*1)
end



%% audit distances
% if show_figure
%     subplot 212
%     plotyy(1:rois,d_sign(ix),1:rois,log10(p_value(ix)))
%     legend('Signed distance,','p value (log_1_0)')
%     %
%     ns=size(Xs,1);
%     no=size(Xo,1);
%     y_temp=repmat('Unrelated',ns+no,1);
%     y_temp(1:ns,:)=repmat('Siblings ',ns,1);
%     up_to=4;
%     for ii=1:up_to
%         subplot(4,up_to,ii)
%         i=ix(ii);
%         boxplot([Xs(:,i); Xo(:,i)],y_temp)
%         ylim([0 1])
%         title(['ROI ' num2str(i)])
%         grid
%
%
%         subplot(4,up_to,ii+up_to)
%         i=ix(end+1-ii);
%         boxplot([Xs(:,i); Xo(:,i)],y_temp)
%         ylim([0 1])
%         title(['ROI ' num2str(i)])
%         grid
%     end
% end
%
