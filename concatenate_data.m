function [X,y] = concatenate_data(Features1,Features2)

X=[Features1;Features2];
y=[ones(size(Features1,1),1);zeros(size(Features2,1),1)];