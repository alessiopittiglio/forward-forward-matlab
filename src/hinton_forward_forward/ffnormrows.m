function [normaliseda] = ffnormrows(a)
%%makes every a have a sum of squared activities that averages 1 per neuron. 

tiny = exp(-100);
numcomp = size(a,2);

normaliseda = a./repmat(tiny + mean(a.^2, 2).^0.5, 1, numcomp);
