function vol = Data_process(path,type,file)
fpath=mfilename('fullpath');
[wpath,~]=fileparts(fpath);
cd(wpath);
addpath(genpath(wpath));

% path: the directory that place the folder namely 'axc' and 't2', which
% contains the dicom image files
% type: the parametric images that want to do the pre-process e.g.
% type={'axc','t2'} or type={'axc'}
% file: filename of nifty file that convert from dicom file e.g. img.nii (p.s. please don't modify)

%% Convert DICOM to NIFTI
fprintf('Checking Nifti files:...\n');
for t=1:length(type)
    k=dir([path '\' type{t} '\*.nii']);    
    if isempty(k)==1
        kk=dir([path '\' type{t}]);
        if length(kk)>3
            Option='-b n -z n -f img';
            eval(['!dcm2niix.exe ' Option ' ' path '\' type{t}]);
            mkdir([path '\' type{t} '\dicom']);
            for di=3:length(kk)
                movefile([path '\' type{t} '\' kk(di).name],[path '\' type{t} '\dicom\' kk(di).name]);
            end
        else
            fprintf('Empty Folder!\n');
        end
    end
end
%% Image pre-process
fprintf('Image pre-process start:...\n');
M4_protocol_pre_GK(path,type,file);
fprintf('Image pre-process has done!!\n');
%% Segment
% fprintf('DL model segmenting...\n');
% if length(type)>1
%     status=system([wpath '\tensorflow\python ' wpath '\M4t1ct2.py -i ' path]);
% end
% if length(type)>2
%     status=system([wpath '\tensorflow\python ' wpath '\M4t1ct2t1.py -i ' path]);
% end
% status=system([wpath '\tensorflow\python ' wpath '\M4t1c.py -i ' path]);
%% Volume calculation
vol=M4_Nhybrid(path,1);
vol=roundn(vol/1000,-4);
fprintf('Tumor volume: %s ml\n',num2str(vol));
% fprintf('DL model segmentation done!!\n');
%%
rmpath(genpath(wpath));
end
