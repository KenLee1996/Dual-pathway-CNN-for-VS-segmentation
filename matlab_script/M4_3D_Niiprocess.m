function M4_3D_Niiprocess(path,path1,pnum)
finfo=niftiinfo(path);
img=niftiread(path);
n=size(img,3);
if n>pnum
    if n>=27 && n<54
        upz=img(:,:,1:pnum);
        fimg=upz;
        finfo.ImageSize(3)=pnum;
    elseif n>=54
        upz=img(:,:,n-48:n-29);
        fimg=upz;
        finfo.ImageSize(3)=pnum;
    else        
        p=abs(pnum-n);
        up=abs(round(p/2));
        upz=img(:,:,up:up+pnum-1);
        fimg=upz;
        finfo.ImageSize(3)=pnum;
    end
elseif n<pnum
    p=pnum-n;
    up=round(p/2);
    upz=zeros(size(img,1),size(img,2),pnum);
    upz(:,:,up:up+n-1)=img;
    fimg=upz;
%     finfo.raw.dim(4)=pnum;
    finfo.ImageSize(3)=pnum;    
else
    upz=zeros(size(img,1),size(img,2),pnum);
    upz(:,:,:)=img;
    fimg=upz;
    finfo.ImageSize(3)=pnum;
end
fimg=int16(fimg);
transform=zeros(4,4);
for pos=1:4
    transform(pos,pos)=finfo.Transform.T(pos,pos);
end
finfo.Transform.T=transform;
finfo.Datatype='int16';
niftiwrite(fimg,path1,finfo);
end
