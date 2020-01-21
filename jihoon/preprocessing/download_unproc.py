from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import os

def download_orig(orig_path, preproc_path):
    matlab_abspath='/usr/local/bin/matlab'
    save_fname = preproc_path.split('/')[-1].split('.')[0]+'_compounds.mat'
    savepath = '/'.join(preproc_path.replace('smooth5','compounds').split('/')[:-1])+'/'
    
    orig_save_path = orig_path.replace('/media/cnir09/GPU1_sync/data/Deep_pain_connectome/HCP_1200/',
                                       '/media/das/Human_Connectome_Project/')
    orig_save_path = orig_save_path.replace('3T/','')
    orig_save_path = '/'.join(orig_save_path.split('/')[:-2])
    orig_fname = orig_path.split('/').pop()
    full_orig_save_path = os.path.join(orig_save_path,orig_fname)
    
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    if not os.path.exists(orig_save_path):
        os.mkdir(orig_save_path)
    
    if not os.path.exists(full_orig_save_path.replace('.gz','')):
        _ = os.system(f'cp {orig_path} {orig_save_path}')
        error = os.system(f'gunzip {full_orig_save_path}')
        
    if not error :
        full_orig_save_path = full_orig_save_path.replace('.gz','')
        
hcp_orig_dataset_root = '/media/das/Human_Connectome_Project/'
unpreproc = sorted(pd.read_csv('unprep_list.csv')['data_path'])
hcp_data = []
hcp_orig = []

for dirpath, _, filenames in os.walk(hcp_orig_dataset_root):
    for fname in filenames:
        fullpath = os.path.join(dirpath,fname)
                
        if 'nii' in fullpath and 'smooth5' in fullpath:
            subj_num = fullpath.split('/')[-3]
            _, TASK, DIRECTION = fullpath.split('/')[-1].split('_')
            DIRECTION = DIRECTION.split('.')[0]

            for origpath in unpreproc:
                if all([subj_num in origpath, TASK in origpath, DIRECTION in origpath]):
                    break
            else:
                break
                    
            hcp_data.append(fullpath)
            hcp_orig.append(origpath.replace('/home/hahnz/CNIR09','/media/cnir09'))
            
sorted_order = sorted(range(len(hcp_data)), key=lambda k: hcp_data[k])
start = 13500
end = len(hcp_data)
hcp_data = [hcp_data[i] for i in sorted_order][start:end]
hcp_orig = [hcp_orig[i] for i in sorted_order][start:end]

mp = Pool(16)
pbar = tqdm(total=len(hcp_data))

def update(result):
    pbar.update(1)
    
for orig, preproc in zip(hcp_orig, hcp_data):
    mp.apply_async(download_orig, args=(orig, preproc,), callback=update)
    
mp.close()
mp.join()
pbar.close()