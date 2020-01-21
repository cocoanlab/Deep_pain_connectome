from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import os

def extract_comp(preproc_path):
    matlab_abspath='/usr/local/bin/matlab'
    save_fname = preproc_path.split('/')[-1].split('.')[0]+'_compounds.mat'
    savepath = '/'.join(preproc_path.replace('smooth5','compounds').split('/')[:-1])+'/'
    
    orig_raw_path = preproc_path.replace('compounds','unprocessed')
    
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    processed = next(os.walk(savepath))[-1]
    command = ['nohup',matlab_abspath,
               '-singleCompThread','-nodisplay','-nodesktop','-r', 
               f'"get_outlier_wm_csf {orig_raw_path} {preproc_path} {savepath} ;exit" >/dev/null 2>&1']
    
    if save_fname not in processed:
        error_occur = os.system(' '.join(command))
        if error_occur:
            print(preproc_path)
            
hcp_orig_dataset_root = '/media/das/Human_Connectome_Project/'
hcp_data = []

for dirpath, _, filenames in os.walk(hcp_orig_dataset_root):
    for fname in filenames:
        fullpath = os.path.join(dirpath,fname)
        orig_path_check = fullpath.replace('smooth5','unprocessed')
        if 'nii' in fullpath and 'smooth5' in fullpath and os.path.exists(orig_path_check):
            hcp_data.append(fullpath)

mp = Pool(20)
pbar = tqdm(total=len(hcp_data))

def update(result):
    pbar.update(1)
    
for path in hcp_data:
    mp.apply_async(extract_comp, args=(path,), callback=update)
    
mp.close()
mp.join()
pbar.close()