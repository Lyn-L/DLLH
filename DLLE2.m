function [B,R] = DLLE2(V,S,b_itr,s_itr,alpha)
bit = size(V,2);
N = size(V,1);
R = randn(bit,bit);
[U11 S2 V2] = svd(R);
R = U11(:,1:bit);
hat_S = S'*S;
%alpha = 2;
% ITQ to find optimal rotation
Z = V * R;
UX = ones(size(Z,1),size(Z,2)).*-1;
UX(Z>=0) = 1;
SDiag = 2*full(diag(hat_S))-4*full(diag(S));

SS = (2*(S+S')-2*hat_S);
for iter=0:b_itr
    tt = 2*alpha*V*R;
    for bb=1:bit
        for ss=1:s_itr
            %tic
            for nn=1:N
%注释掉的完全按照公式来写，但是这样速度太慢，有太多的重复计算，下面的运算速度快
%                  u=UX(:,bb)'*(2*S(:,nn)+2*S(nn,:)'-hat_S(:,nn)-hat_S(nn,:)')+2*alpha*V(nn,:)*R(:,bb)
%                  +UX(nn,bb)*(2*hat_S(nn,nn)-4*S(nn,nn));
%                  Z(nn,bb)=u;
                u=UX(:,bb)'*SS(:,nn)+tt(nn,bb)+UX(nn,bb)*SDiag(nn);
                Z(nn,bb)=u;
            end
            %toc
        end
    end
    UX(:) =-1;
    UX(Z>=0) = 1;
    C = UX' * V;
    [UB,sigma,UA] = svd(C);    
    R = UA * UB';
    Z = V * R;
    UX(:) =-1;
    UX(Z>=0) = 1;
end
% make B binary
Z = V * R;
UX(:) =-1;
UX(Z>=0) = 1;
B = UX;
B(B<0) = 0;   
end