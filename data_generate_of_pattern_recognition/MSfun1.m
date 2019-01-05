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
YY=[ZZ(:,1:2:end-1),ZZ(:,2:2:end),skewness(XX,1,2),kurtosis(XX,1,2),sum(XX.^2,2)/(kk*nn),SB,AASL,SRANGE,REAE];
 else
 error('Wrong format')
 end
end
function tmse=msefun(XX)
 YY=XX-repmat(mean(XX,2),1,size(XX,2));
 tmse=sqrt(sum(YY.^2,2));
end