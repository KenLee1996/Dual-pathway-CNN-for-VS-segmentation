function M4_depadding(path,pnum,num)
% num=1;
fimg=niftiread([path '\axc\mmimg.nii']);
finfo=niftiinfo([path '\axc\mmimg.nii']);
mimg=niftiread([path '\mtv' num2str(num) '.nii']);
% mimg=niftiread([path '\axc\tv120.nii']);
% finfo=niftiinfo([path '\axc\tv120.nii']);
% minfo=load_untouch_nii([path '\mtv.nii']);

img=fimg;
n=size(img,3);
tmp=zeros(size(img,1),size(img,2),size(img,3));
if n>pnum
    if n>=27 && n<54        
        tmp(:,:,1:pnum)=mimg;
    elseif n>=54
        tmp(:,:,n-48:n-29)=mimg;
    else
        p=abs(pnum-n);
        up=abs(round(p/2));
        tmp(:,:,up:up+pnum-1)=mimg;
    end
elseif n<pnum
    p=pnum-n;
    up=round(p/2);
    tmp=mimg(:,:,up:up+n-1);
else
    tmp=mimg;    
end
finfo.Datatype='int16';
transform=zeros(4,4);
for pos=1:4
    transform(pos,pos)=finfo.Transform.T(pos,pos);
end
finfo.Transform.T=transform;
tmp=int16(tmp);
niftiwrite(tmp,[path '\tv' num2str(num) '.nii'],finfo);
end











