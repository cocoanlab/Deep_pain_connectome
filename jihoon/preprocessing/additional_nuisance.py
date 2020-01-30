import os
import nibabel as nib
from nilearn.image import clean_img
from scipy.io import loadmat
from multiprocessing import Pool
from tqdm import tqdm

hcp_orig_dataset_root = '/media/das/Human_Connectome_Project/'
hcp_data = []
hcp_nuisance = []
def highpass_filtering(fmripath, confound_path, Hz=.008, TR=0.72):
    matlab_abspath='/usr/local/bin/matlab'
    savepath = fmripath.replace('smooth5', 'hpf_result')
    savedir, save_fname = os.path.split(savepath)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
        
    processed = next(os.walk(savedir))[-1]
    command = ['nohup',matlab_abspath,
               '-singleCompThread','-nodisplay','-nodesktop','-r', 
               f'"remove_confounds_HPF {fmripath} {confound_path} {savepath} ;exit" >/dev/null 2>&1']
    
    if save_fname not in processed:
        error_occur = os.system(' '.join(command))
        if error_occur:
            print(preproc_path)

'''[python] Nilearn Highpass filtering with removing confounds
def highpass_filtering(fmripath, confounds_path, detrend=False, standardize=True, Hz=.008, TR=0.72):
    savepath = fmripath.replace('smooth5', 'hpf_result')
    savedir, _ = os.path.split(savepath)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
        
    smoothed = nib.load(fmripath)
    nuisance_confounds = loadmat(confounds_path)['compounds']
    
    regressed = clean_img(smoothed, detrend=detrend, standardize=standardize, confounds=nuisance_confounds, high_pass=Hz, t_r=TR)
    nib.save(regressed, savepath)
    if smoothed.in_memory : smoothed.uncache()
    if regressed.in_memory : regressed.uncache()
    del smoothed, regressed 
'''
for dirpath, _, filenames in os.walk(hcp_orig_dataset_root):
    for fname in filenames:
        fullpath = os.path.join(dirpath,fname)
        nuisance_path = fullpath.replace('smooth5','compounds').replace('.nii','_nuisance.mat')
        
        if 'nii' in fullpath and 'smooth5' in fullpath and os.path.exists(nuisance_path):
            hcp_data.append(fullpath)
            hcp_nuisance.append(nuisance_path)
            

mp = Pool(20)
pbar = tqdm(total=len(hcp_data))
pbar.set_description('[ HPF ] : ')
def update(result):
    pbar.update(1)
    
for fmripath, confounds_path in zip(hcp_data, hcp_nuisance):
    mp.apply_async(highpass_filtering, args=(fmripath, confounds_path,), callback=update)
    
mp.close()
mp.join()
pbar.close()