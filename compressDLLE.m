function [Y] = compressDLLE(XX, pc, R)

XX = XX*pc*R;
Y = zeros(size(XX));
Y(XX>=0) = 1;
Y = compactbit(Y>0);
end