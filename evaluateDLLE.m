function evaluation_info = evaluateDLLE(data, param)
bit = param.nbits;

groundtruth = data.groundtruth;
traindata = data.train_data';
dbdata = data.db_data';
tstdata = data.test_data';

sampleMean = mean(traindata,1);
traindata = (double(traindata)-repmat(sampleMean,size(traindata,1),1));
dbdata = (double(dbdata)-repmat(sampleMean,size(dbdata,1),1));
tstdata = (double(tstdata)-repmat(sampleMean,size(tstdata,1),1));
traindata = normalize1(traindata);
dbdata = normalize1(dbdata);
tstdata = normalize1(tstdata);

b_itr = param.b_itr;
s_itr = param.s_itr;
alpha = param.alpha;
K     = param.K; 


timerTrain = tic;
[pc, l] = eigs(cov(traindata),bit);
S=getSimilarMatrix(traindata',K);
traindata = traindata * pc;
[~,R]=DLLE2(traindata,S,b_itr,s_itr,alpha);
trainT=toc(timerTrain);

[B_db] = compressDLLE(dbdata, pc, R);
% B_db = OneLayerAGH_Test(dbdata, centers, W, s, sigma);
if(isfield(data, 'groundtruth'))
    
    timeTest = tic;
    [B_tst] = compressDLLE(tstdata, pc, R);
%     B_tst = OneLayerAGH_Test(tstdata, centers, W, s, sigma);
    compressT=toc(timeTest)/size(B_tst,1);
    
    evaluation_info = performance(B_tst, B_db, groundtruth, param);
    evaluation_info.trainT = trainT;
    evaluation_info.compressT = compressT;
else
    D_dist =  -hammingDist(B_db,B_db);
    evaluation_info.AP = compute_avg_top(D_dist);
end


end