function [regressout] = remove_confounds_HPF(preproc_path, confounds_path, save_path)

addpath(genpath(pwd));
addpath(genpath('/usr/local/Resources/spm12'));

tfmri = fmri_data(preproc_path);
load(confounds_path);
tfmri.covariates = confounds;

regressout = canlab_connectivity_preproc(tfmri, 'hpf', .008, 0.72, 'windsorize', 5);

write(regressout, 'fname', save_path)
