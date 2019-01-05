clear;
clc;
echo off
%Part1:generate data
if exist('data.mat','file')
load('data.mat')
else

NN=60;
MM=1000;
% pattern1: Upward trend
XX1=0.5*randn(MM,NN)+(0.05+0.05*rand(MM,NN)).*[ones(MM,NN/2),repmat((1:1:NN/2),MM,1)]; 
XX1=MSfun1(XX1,10);
YY1=repmat([1,0,0,0,0],MM,1);
% pattern2:Downward trend
XX2=0.5*randn(MM,NN)-(0.05+0.05*rand(MM,NN)).*[ones(MM,NN/2),repmat((1:1:NN/2),MM,1)];
XX2=MSfun1(XX2,10);
YY2=repmat([0,1,0,0,0],MM,1);
% pattern3:Up step
XX3=0.5*randn(MM,NN)+(0.8+1.5*rand(MM,NN)).*[zeros(MM,NN/2),ones(MM,NN/2)];
XX3=MSfun1(XX3,10);
YY3=repmat([0,0,1,0,0],MM,1);
% pattern4:down step
XX4=0.5*randn(MM,NN)-(0.8+1.5*rand(MM,NN)).*[zeros(MM,NN/2),ones(MM,NN/2)];

XX4=MSfun1(XX4,10);
YY4=repmat([0,0,0,1,0],MM,1);
% pattern5:normal
XX5=0.5*randn(MM,NN);
XX5=MSfun1(XX5,10);
YY5=repmat([0,0,0,0,1],MM,1);
save data1.mat XX1 XX2 XX3 XX4 XX5 YY1 YY2 YY3 YY4 YY5
%Part2:split train data and test data
datainptrain=[XX1(1:(MM/6),:);XX2(1:(MM/6),:);XX3(1:(MM/6),:);XX4(1:(MM/6),:);XX5(1:(MM/6),:)];
dataouttrain=[YY1(1:(MM/6),:);YY2(1:(MM/6),:);YY3(1:(MM/6),:);YY4(1:(MM/6),:);YY5(1:(MM/6),:)];
datainptest=[XX1(((MM/6)+1):MM,:);XX2(((MM/6)+1):MM,:);XX3(((MM/6)+1):MM,:);XX4(((MM/6)+1):MM,:);XX5(((MM/6)+1):MM,:)];
dataouttest=[YY1(((MM/6)+1):MM,:);YY2(((MM/6)+1):MM,:);YY3(((MM/6)+1):MM,:);YY4(((MM/6)+1):MM,:);YY5(((MM/6)+1):MM,:)];
save data.mat datainptrain dataouttrain datainptest dataouttest
end

%Part3:BP NN
inpmm=[min(min(datainptrain)),max(max(datainptrain))];
bpinp=(datainptrain.'-inpmm(1))/(inpmm(2)-inpmm(1));
otpmm=[min(min(dataouttrain)),max(max(dataouttrain))];
bpout=(dataouttrain.'-otpmm(1))/(otpmm(2)-otpmm(1));
bpnet=newff(bpinp,bpout,[10,7],{'logsig', 'logsig'}, 'trainlm', 'learngd');
bpnet.trainParam.epochs=1000;
bpnet.trainParam.goal=0.01;
bpnet.trainParam.show=100;
bpnet.trainParam.lr=0.01;
bpnet=init(bpnet);
bpnet=train(bpnet,bpinp,bpout);
save bpnet.mat bpnet inpmm otpmm
bptraincheck = sim(bpnet,bpinp); 
bptraincheck = bptraincheck*(otpmm(2)-otpmm(1))+otpmm(1);
bptraincheck = full(compet(bptraincheck));
bptraintest = sim(bpnet,(datainptest.'-inpmm(1))/(inpmm(2)-inpmm(1)));
bptraintest = bptraintest*(otpmm(2)-otpmm(1))+otpmm(1);
bptraintest = full(compet(bptraintest));

train2=CMPfun(bpout,bptraincheck)
test2=CMPfun(dataouttest',bptraintest)