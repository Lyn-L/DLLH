function distAnchor(dataset,bit,b_itr,s_itr,neis,alphas,simpleNum)
AnchorNum = 300;
switch(dataset)
    case 'cifar'
        load(dataset);X=cifar10(randperm(60000,simpleNum),1:320);%test(X,32,'DLLE');test(X,32,'ITQ');
    case 'mnist_data'
        load(dataset);X=traindata(randperm(69000,simpleNum),:);%test(X,32,'DLLE');test(X,32,'ITQ');
    case 'LM_data'
        load(dataset);X=X(randperm(22019,simpleNum),:);%test(X,32,'DLLE');test(X,32,'ITQ');
    case 'nusy'
        load(dataset);X=testImg(randperm(116000,simpleNum),:);%test(X,32,'DLLE');test(X,32,'ITQ');
    case 'usps'
        load(dataset);X=samples';%test(X,32,'DLLE');test(X,32,'ITQ');
end
Anchor = X(randperm(size(X,1),AnchorNum),:);
%['b_',num2str(b_itr),'s_',num2str(s_itr),'smplenum',num2str(simpleNum)]
distTest(X,bit,'ITQ',dataset,['b_',num2str(b_itr),'s_',num2str(s_itr),'smplenum',num2str(simpleNum),'anchor']);
averageNumberNeighbors = 50;    % ground truth is 50 nearest neighbor
num_test = 1000;                % 1000 query test point, rest are database
% split up into training and test set
[ndata, D] = size(X);
R = randperm(ndata);
Xtest = X(R(1:num_test),:);
R(1:num_test) = [];
Xtraining = X(R,:);
num_training = size(Xtraining,1);
clear X;
% define ground-truth neighbors (this is only used for the evaluation):
R = randperm(num_training);
DtrueTraining = distMat(Xtraining(R(1:100),:),Xtraining); % sample 100 points to find a threshold
Dball = sort(DtrueTraining,2);
clear DtrueTraining;
Dball = mean(Dball(:,averageNumberNeighbors));
% scale data so that the target distance is 1
Xtraining = Xtraining / Dball;
Xtest = Xtest / Dball;
Dball = 1;
% threshold to define ground truth
DtrueTestTraining = distMat(Xtest,Xtraining);
WtrueTestTraining = DtrueTestTraining < Dball;
clear DtrueTestTraining
% generate training ans test split and the data matrix
XX = [Xtraining; Xtest];
% center the data, VERY IMPORTANT
sampleMean = mean(XX,1);
XX = (XX - repmat(sampleMean,size(XX,1),1));
% PCA
[pc, l] = eigs(cov(XX(1:num_training,:)),bit);
XX = XX * pc;

ZZ = XX;
Anchor = Anchor*pc;
for nei=neis
    S = getAnchorW(ZZ(1:num_training,:),Anchor,nei);
    size(S)
    for alpha=alphas
        size(S)
        [Y, R]  = DLLE(ZZ(1:num_training,:),S,b_itr,s_itr,alpha);
        XX = ZZ*R;
        Y = zeros(size(XX));
        Y(XX>=0) = 1;
        Y = compactbit(Y>0);
        B1 = Y(1:size(Xtraining,1),:);
        B2 = Y(size(Xtraining,1)+1:end,:);
        Dhamm = hammingDist(B2, B1);
        size(Dhamm)
        [recall, precision, rate] = recall_precision(WtrueTestTraining, Dhamm);
        save(['distance/',num2str(bit),'DLLE',dataset,num2str(nei),num2str(alpha*10),'b_',num2str(b_itr),'s_',num2str(s_itr),'smplenum',num2str(simpleNum),'anchor'],'recall','precision','alpha','nei','R');
        getmap(precision,recall)
    end
end



