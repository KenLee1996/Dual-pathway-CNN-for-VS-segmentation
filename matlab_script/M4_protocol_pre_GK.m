function M4_protocol_pre_GK(path,type,file)
%% Artifact correct
fprintf('Rescale and Artifact correct start:...\n');
for t=1:length(type)
    minfo=niftiinfo([path '\' type{t} '\' file]);
    mimg=niftiread([path '\' type{t} '\' file]);
    mimg=mimg+minfo.AdditiveOffset;
    minfo.AdditiveOffset=0;
    mimg=int16(mimg);
    transform=zeros(4,4);
    for pos=1:4
        transform(pos,pos)=minfo.Transform.T(pos,pos);
    end
    minfo.Transform.T=transform;
    minfo.Datatype='int16';
    if minfo.raw.pixdim(1)==1
        minfo.raw.pixdim(1)=-1;
        mimg=flipud(mimg);
    end
    if minfo.Transform.T(1,1)>0
       minfo.Transform.T(1,1)=minfo.Transform.T(1,1)*(-1);        
    end
    if minfo.Transform.T(2,2)<0
       minfo.Transform.T(2,2)=minfo.Transform.T(2,2)*(-1);        
    end
    niftiwrite(mimg,[path '\' type{t} '\i' file],minfo);
    mimg(find(mimg<0))=0;
    tt=tabulate(mimg(:));
    value=tt(:,1);
    out=value(find(isoutlier(value(:))==1));
    if isempty(out)==0
        min_out_value=min(out);
        ind=find(value==min_out_value);
        cor_value=value(ind-1);
        mimg(find(mimg>=cor_value))=cor_value;
        mimg=hi_cor(mimg,1);
    else
        mimg=hi_cor(mimg,0);
    end
    mimg=int16(mimg);
    minfo.Datatype='int16';
    transform=zeros(4,4);
    for pos=1:4
        transform(pos,pos)=minfo.Transform.T(pos,pos);
    end
    minfo.Transform.T=transform;
    niftiwrite(mimg,[path '\' type{t} '\m' file],minfo);
    
end

    function nimg=hi_cor(img,n)
        hitt=tabulate(img(:));
        hitt=hitt(1:end-n,:);
        hs=0;
        for j=length(hitt):-1:1
            if hs==0
                if hitt(j,2)>80
                    hs=hitt(j,1);
                end
            else
                break
            end
        end
        img(find(img>=hs))=hs;
        nimg=img;
    end
fprintf('Rescale and Artifact correct:...done\n');
%% Mask
fprintf('Creating Mask:...\n');
t=1;
finfo=niftiinfo([path '\' type{t} '\m' file]);
fimg=niftiread([path '\' type{t} '\m' file]);
tmp=fimg;
level = graythresh(tmp);
BW = imbinarize(tmp,level);
b=zeros(size(tmp,1),size(tmp,2),size(tmp,3));
se = strel('sphere',3);
BW=imdilate(BW,se);
for i=1:size(tmp,3)
    b(:,:,i)=imfill(BW(:,:,i),'holes');
end
nb=zeros(size(tmp,1),size(tmp,2),size(tmp,3));
for i=1:size(tmp,3)
    nb(:,:,i)=bwareaopen(b(:,:,i),10000);
end
transform=zeros(4,4);
    for pos=1:4
        transform(pos,pos)=finfo.Transform.T(pos,pos);
    end
finfo.Transform.T=transform;
fimg=int16(double(fimg).*nb);
finfo.Datatype='int16';
niftiwrite(fimg,[path '\' type{t} '\mm' file],finfo);
fimg=int16(nb);
niftiwrite(fimg,[path '\' type{t} '\mask.nii'],finfo);
fprintf('Creating Mask:...done\n');
%% Registration
if length(type)>1
    for t=2:length(type)
        fprintf(['Registration from ' type{t} ' to ' type{1} ':...\n']);
        ref=[path '\' type{1} '\mm' file ',1'];
        source=[path '\' type{t} '\m' file ',1'];
%         other=[path '\' type{t} '\tv1,1'];
        matlabbatch{t-1}.spm.spatial.coreg.estwrite.ref = {ref};
        matlabbatch{t-1}.spm.spatial.coreg.estwrite.source = {source};
        matlabbatch{t-1}.spm.spatial.coreg.estwrite.other = {''};
        matlabbatch{t-1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
        matlabbatch{t-1}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
        matlabbatch{t-1}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
        matlabbatch{t-1}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
        matlabbatch{t-1}.spm.spatial.coreg.estwrite.roptions.interp = 6;
        matlabbatch{t-1}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
        matlabbatch{t-1}.spm.spatial.coreg.estwrite.roptions.mask = 0;
        matlabbatch{t-1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';
    end
    job=matlabbatch;
    spm_jobman('run',job)
    ref_img=niftiread([path '\' type{1} '\mm' file]);
    for t=2:length(type)        
        info=niftiinfo([path '\' type{1} '\mm' file]);
        img1=niftiread([path '\' type{t} '\rm' file]);
        info1=niftiinfo([path '\' type{t} '\rm' file]);
        info1.PixelDimensions=info.PixelDimensions;        
        img1(find(img1<0))=0;
        niftiwrite(img1,[path '\' type{t} '\rm' file],info1);
        
        info=niftiinfo([path '\' type{t} '\m' file]);
        img=niftiread([path '\' type{t} '\m' file]);
        transform=zeros(4,4);
        for pos=1:4
            transform(pos,pos)=info.Transform.T(pos,pos);
        end
        info.Transform.T=transform;
        niftiwrite(img,[path '\' type{t} '\m' file],info);
        ncc=[];
%         tmp=normxcorr3(ref_img,img1,'same');
%         ncc(1,t-1)=max(tmp(:));
    end
end
fprintf('Registration:...done\n');
%% Apply mask
fprintf('Creating Mask:...\n');
for t=2:length(type)
    finfo=niftiinfo([path '\' type{t} '\rm' file]);
    fimg=niftiread([path '\' type{t} '\rm' file]);
    nb=niftiread([path '\' type{1} '\mask.nii']);    
    fimg=int16(double(fimg).*double(nb));
    finfo.Datatype='int16';
    niftiwrite(fimg,[path '\' type{t} '\rmm' file],finfo);
end
fprintf('Creating Mask:...done\n');
%% Padding
pnum=20;
fprintf(['Padding to ' num2str(pnum) ':...\n']);
for t=1:length(type)
    if t==1
        M4_3D_Niiprocess([path '\' type{t} '\mm' file],[path '\' type{t} '\mm'  num2str(pnum) file],pnum);
%         M4_3D_Niiprocess([path '\' type{t} '\tv1.nii'],[path '\' type{t} '\tv1'  num2str(pnum) '.nii'],pnum);
    else
        M4_3D_Niiprocess([path '\' type{t} '\m' file],[path '\' type{t} '\m'  num2str(pnum) file],pnum);        
%         M4_3D_Niiprocess([path '\' type{t} '\rmm' file],[path '\' type{t} '\rmm' num2str(pnum) file],pnum);
%         M4_3D_Niiprocess([path '\' type{t} '\tv1.nii'],[path '\' type{t} '\rtv1'  num2str(pnum) '.nii'],pnum);
    end
end
fprintf('Padding:...done\n');
%% Crop
fprintf('Crop to 256x216x20:...\n');
minfo=niftiinfo([path '\' type{1} '\mm' num2str(pnum) file]);
mimg=niftiread([path '\' type{1} '\mm' num2str(pnum) file]);

u=1;
d=size(mimg,1);
l=1;
r=size(mimg,2);
for ind=1:size(mimg,1)
    s=sum(mimg(ind,:,:));
    if sum(s)>0
        u=ind;
        break
    end
end
for ind=size(mimg,1):-1:1
    s=sum(mimg(ind,:,:));
    if sum(s)>0
        d=ind;
        break
    end
end
for ind=1:size(mimg,2)
    s=sum(mimg(:,ind,:));
    if sum(s)>0
        l=ind;
        break
    end
end
for ind=size(mimg,2):-1:1
    s=sum(mimg(:,ind,:));
    if sum(s)>0
        r=ind;
        break
    end
end
o=[round((u+d)/2),round((l+r)/2)];
mimg=mimg((o(1)-127):(o(1)+128),(o(2)-107):(o(2)+108),:);
minfo.ImageSize(1)=256;
minfo.ImageSize(2)=216;
transform=zeros(4,4);
for pos=1:4
    transform(pos,pos)=minfo.Transform.T(pos,pos);
end
minfo.Transform.T=transform;
niftiwrite(mimg,[path '\' type{1} '\cmm' num2str(pnum) file],minfo);

% if length(type)>1
%     for t=1:length(type)

% t=1;
% if t==1
%     minfo=niftiinfo([path '\' type{t} '\tv1' num2str(pnum) '.nii']);
%     mimg=niftiread([path '\' type{t} '\tv1' num2str(pnum) '.nii']);
%     mimg=mimg((o(1)-127):(o(1)+128),(o(2)-107):(o(2)+108),:);
%     minfo.ImageSize(1)=256;
%     minfo.ImageSize(2)=216;
%     transform=zeros(4,4);
%     for pos=1:4
%         transform(pos,pos)=minfo.Transform.T(pos,pos);
%     end
%     minfo.Transform.T=transform;
%     niftiwrite(mimg,[path '\' type{t} '\ctv1' num2str(pnum) '.nii'],minfo);
% else
%     minfo=niftiinfo([path '\' type{t} '\rtv1' num2str(pnum) '.nii']);
%     mimg=niftiread([path '\' type{t} '\rtv1' num2str(pnum) '.nii']);
%     mimg=mimg((o(1)-127):(o(1)+128),(o(2)-107):(o(2)+108),:);
%     minfo.ImageSize(1)=256;
%     minfo.ImageSize(2)=216;
%     transform=zeros(4,4);
%     for pos=1:4
%         transform(pos,pos)=minfo.Transform.T(pos,pos);
%     end
%     minfo.Transform.T=transform;
%     niftiwrite(mimg,[path '\' type{t} '\crtv1' num2str(pnum) '.nii'],minfo);
% end
%     end
% end

if length(type)>1
    for t=2:length(type)
        minfo=niftiinfo([path '\' type{t} '\rmm' num2str(pnum) file]);
        mimg=niftiread([path '\' type{t} '\rmm' num2str(pnum) file]);
        mimg=mimg((o(1)-127):(o(1)+128),(o(2)-107):(o(2)+108),:);
        minfo.ImageSize(1)=256;
        minfo.ImageSize(2)=216;
        transform=zeros(4,4);
        for pos=1:4
            transform(pos,pos)=minfo.Transform.T(pos,pos);
        end
        minfo.Transform.T=transform;
        niftiwrite(mimg,[path '\' type{t} '\crmm' num2str(pnum) file],minfo);
    end
end
save([path '\' type{1} '\cropping orientation.mat'],'o');
fprintf('Cropping:...done\n');
%%
fprintf('Image pre-process has done!!\n');
end
