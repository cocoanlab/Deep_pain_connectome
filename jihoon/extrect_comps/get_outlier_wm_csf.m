function [outlier, wm, csf] = get_outlier_wm_csf(fmripath, savepath)

addpath(genpath(pwd));
addpath(genpath('/home/hahnz/Projects/Resources/spm12'));

tfmri = fmri_data(fmripath);

[~, components] = extract_gray_white_csf(tfmri, 'masks', {which('gray_matter_mask.nii'),...
which('canonical_white_matter_thrp5_ero1.nii'), ...
which('canonical_ventricles_thrp5_ero1.nii')});

outlier = preprocess(tfmri, 'outliers_rmssd').covariates;
wm = scale(double(components{2}));
csf = scale(double(components{3}));

fname = split(fmripath, '/');
fname = split(fname{end}, '.');
fname = fname{1}+"_compounds.mat";

save(savepath+fname);
clear tfmri components fmripath fname savepath;
return;
