clear;
clc;
echo off
%Part1:【仿真数据的产生】
if exist('data.mat','file')
load('data.mat')
else

NN=60;
MM=1000;
% 模式一：上升趋势
XX1=0.5*randn(MM,NN)+(0.05+0.05*rand(MM,NN)).*[ones(MM,NN/2),repmat((1:1:N
N/2),MM,1)]; XX1=MSfun1(XX1,10);
YY1=repmat([1,0,0,0,0],MM,1);
% 模式二：下降趋势
XX2=0.5*randn(MM,NN)-(0.05+0.05*rand(MM,NN)).*[ones(MM,NN/2),repmat((1:1:N
N/2),MM,1)];
XX2=MSfun1(XX2,10);
YY2=repmat([0,1,0,0,0],MM,1);
% 模式三：向上阶跃
XX3=0.5*randn(MM,NN)+(0.8+1.5*rand(MM,NN)).*[zeros(MM,NN/2),ones(MM,NN/2
)];
XX3=MSfun1(XX3,10);
YY3=repmat([0,0,1,0,0],MM,1);
% 模式四：向下阶跃
XX4=0.5*randn(MM,NN)-(0.8+1.5*rand(MM,NN)).*[zeros(MM,NN/2),ones(MM,NN/2)];

XX4=MSfun1(XX4,10);
YY4=repmat([0,0,0,1,0],MM,1);
% 模式五：正常模式
XX5=0.5*randn(MM,NN);
XX5=MSfun1(XX5,10);
YY5=repmat([0,0,0,0,1],MM,1);
save data1.mat XX1 XX2 XX3 XX4 XX5 YY1 YY2 YY3 YY4 YY5
%Part2:【划分训练和测试样本】
datainptrain=[XX1(1:(MM/6),:);XX2(1:(MM/6),:);XX3(1:(MM/6),:);XX4(1:(MM/6),:);X
X5(1:(MM/6),:)];
dataouttrain=[YY1(1:(MM/6),:);YY2(1:(MM/6),:);YY3(1:(MM/6),:);YY4(1:(MM/6),:);Y
Y5(1:(MM/6),:)];
datainptest=[XX1(((MM/6)+1):MM,:);XX2(((MM/6)+1):MM,:);XX3(((MM/6)+1):MM,:)
;XX4(((MM/6)+1):MM,:);XX5(((MM/6)+1):MM,:)];
dataouttest=[YY1(((MM/6)+1):MM,:);YY2(((MM/6)+1):MM,:);YY3(((MM/6)+1):MM,:)
;YY4(((MM/6)+1):MM,:);YY5(((MM/6)+1):MM,:)];
save data.mat datainptrain dataouttrain datainptest dataouttest
end
%Part3:【BP 神经网络】
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



function YY=MSfun1(XX,nn)
 Xsize=size(XX);
 if (rem(Xsize(2),4)==0 && rem(Xsize(2),nn)==0 )
 tt=(1:Xsize(2))-(1+Xsize(2))/2;
 SB=sign(sum(XX.*repmat(tt,Xsize(1),1),2)/sum(tt.^2));
 YY=mat2cell(XX,Xsize(1),Xsize(2)/4*ones(1,4));
 tt=mat2cell(1:Xsize(2),1,Xsize(2)/4*ones(1,4));
 MPnt=cell(2,4);
 for ii=1:1:4
 MPnt{1,ii}=mean(tt{ii});
 MPnt{2,ii}=mean(YY{ii},2);
 end
 sjk=zeros(Xsize(1),6);
 MSEjk=zeros(Xsize(1),6);
 ii=1;
 for jj=1:1:3
 for kk=(jj+1):1:4
 sjk(:,ii)=(MPnt{2,jj}-MPnt{2,kk})/(MPnt{1,jj}-MPnt{1,kk});
 MSEjk(:,ii)=msefun([YY{jj},YY{kk}]);
 ii=ii+1;
 end
 end
 AASL=abs(mean(sjk,2));
 SRANGE=max(sjk,[],2)-min(sjk,[],2);
 REAE=msefun(sjk)./(sum(MSEjk,2)/6);
 
 kk=Xsize(2)/nn;
 YY=mat2cell(XX,Xsize(1),kk*ones(1,nn));
 ZZ=zeros(Xsize(1),2*kk);
 for ii=1:1:kk
 ZZ(:,(2*ii-1):(2*ii))=[mean(YY{ii},2),std(YY{ii},0,2)];
 end
YY=[ZZ(:,1:2:end-1),ZZ(:,2:2:end),skewness(XX,1,2),kurtosis(XX,1,2),sum(X
X.^2,2)/(kk*nn),SB,AASL,SRANGE,REAE];
 else
 error('Wrong format')
 end
end


function tmse=msefun(XX)
 YY=XX-repmat(mean(XX,2),1,size(XX,2));
 tmse=sqrt(sum(YY.^2,2));
end

%模式混淆矩阵
function YY=CMPfun(InpD,OutD)
 mdnm=size(InpD,1);
 YY=zeros(mdnm,mdnm);
 [~,InpN]=max(InpD);
 [~,OutN]=max(OutD);
 for ii=1:1:length(InpN)
 YY(InpN(ii),OutN(ii))=YY(InpN(ii),OutN(ii))+1;
 end
 YY=YY./repmat(sum(YY,2),1,mdnm);
end