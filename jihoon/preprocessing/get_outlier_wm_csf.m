function [outlier, wm, csf] = get_outlier_wm_csf(orig_path, preproc_path, save_path)

maxNumCompThreads = 1;

addpath(genpath(pwd));
addpath(genpath('/home/hahnz/Resources/spm12'));

orig = fmri_data(orig_path);
tfmri = fmri_data(preproc_path);

[~, components] = extract_gray_white_csf(tfmri, 'masks', {which('gray_matter_mask.img'),...
which('canonical_white_matter_thrp5_ero1.nii'), ...
which('canonical_ventricles_thrp5_ero1.nii')});

outlier = preprocess(orig, 'outliers_rmssd').covariates;
wm = scale(double(components{2}));
csf = scale(double(components{3}));

fname = split(preproc_path, '/');
fname = split(fname{end}, '.');
fname = fname{1}+"_compounds.mat";

clear orig tfmri components preproc_path orig_path;
save(save_path+fname);
