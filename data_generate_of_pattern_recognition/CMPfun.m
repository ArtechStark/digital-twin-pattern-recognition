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