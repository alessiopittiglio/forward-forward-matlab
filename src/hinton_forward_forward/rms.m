function [rootmeansquare] = rms(x)
%%assumes x is a matrix, but should work for vectors and scalars too.

rootmeansquare = sqrt(sum(sum(x.^2))/(size(x,1)*size(x,2)));

