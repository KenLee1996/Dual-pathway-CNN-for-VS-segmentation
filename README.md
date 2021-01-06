# Dual-pathway-CNN-for-VS-segmentation
 <h2> Software platform for data pre-processing</h2> 
 <p> Matlab 2019b+ </p>
 <p> matlab_script/Data_process.m: Main matlab script for pre-processing</p>
 <p> matlab_script/Volume_calculation.m: Volume calculation after CNN inference</p> 
 
 <h2> Software platform for CNN training</h2> 
 <p> python3.6, tensorflow-gpu==1.15, keras==2.3.1 </p>
 
 <h2> IPython notebook </h2>
 <p> Main.ipynb: Model defined, training and inference </p>
 
 <h2> Example pipeline for tumor volume evaluation</h2> 
 <p> 1.Data preparation: MR Axial Spin-Echo T1W+C dicom files at './subjec1/axc/' </p>
 <p> 2.Data preparation: MR Axial Spin-Echo T2W dicom files at './subjec1/t2/' </p>
 <p> 3.Matlab pre-processing: Data_process('./subjec1/',{'axc','t2'},'img.nii') </p>
 <p> 4.IPython Notebook model inference: Predict the tumor probability map using fourth cell of Main.ipynb </p>
 <p> 5.Matlab volume evaluation: Volume_calculation('./subjec1/') </p>
 <p> *MR: Magnetic Resonance </p>
 
 <h2> Publication</h2>
 <p> https://www.sciencedirect.com/science/article/pii/S0933365719312722 </p>
