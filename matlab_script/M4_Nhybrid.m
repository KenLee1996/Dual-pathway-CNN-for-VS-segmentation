function v=M4_Nhybrid(path,num)
% num=1;
pnum=20;
m=niftiread([path '\predicted' num2str(num) '_tv.nii']);
finfo=niftiinfo([path '\predicted' num2str(num) '_tv.nii']);
o=double(m);
o(find(o(:)<0.5))=0;
o(:,1:(size(o,2)/8),:)=0;
o(:,(size(o,2)*7/8):end,:)=0;
o(find(o(:)>=0.5))=1;
if length(find(o(:)==1))~=0
    p=select_group(o);
    v=sum(o(:))*finfo.raw.pixdim(2)*finfo.raw.pixdim(3)*finfo.raw.pixdim(4);
else
    p=zeros(size(m,1),size(m,2),size(m,3));
    v=0;
end

fimg=niftiread([path '\axc\mm20img.nii']);
finfo=niftiinfo([path '\axc\mm20img.nii']);
load([path '\axc\cropping orientation.mat']);
% p=M4_tvconvex_hull(p);
tmp=zeros(size(fimg,1),size(fimg,2),size(fimg,3));
tmp((o(1)-127):(o(1)+128),(o(2)-107):(o(2)+108),:)=p;

fimg=int16(tmp);
finfo.Datatype='int16';
transform=zeros(4,4);
for pos=1:4
    transform(pos,pos)=finfo.Transform.T(pos,pos);
end
finfo.Transform.T=transform;
niftiwrite(fimg,[path '\mtv' num2str(num) '.nii'],finfo);
M4_depadding(path,20,num);
delete([path '\mtv' num2str(num) '.nii']);
end
function no=select_group(o)
CC=bwconncomp(o, 26);
S=regionprops(CC, 'Area');
L=labelmatrix(CC);
c=L;
t=tabulate(c(:));
% mL=extractfield(S,'Area');
% ind=find(mL==max(mL));
a=t(2:end,2);
ind=find(a==max(a));
if length(ind)>1
    L=ismember(L,t(ind+1,1));
elseif length(ind)==1
    L(find(L~=t(ind+1,1)))=0;
    L(find(L==t(ind+1,1)))=1;
end
o=L;
se=strel('sphere',3);
for h=1:size(o,3)
    o(:,:,h)=imfill(o(:,:,h),'holes');
    o(:,:,h)=bwareaopen(o(:,:,h),10);
end
no=o;
end
