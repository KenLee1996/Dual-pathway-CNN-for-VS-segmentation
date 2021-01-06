function vol=Volume_calculation(path)
vol=M4_Nhybrid(path,1);
vol=roundn(vol/1000,-4);
fprintf('Tumor volume: %s ml\n',num2str(vol));