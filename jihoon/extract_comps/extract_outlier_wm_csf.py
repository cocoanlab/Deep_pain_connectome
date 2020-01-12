from multiprocessing import Pool
from tqdm import tqdm
import os

def extract_comp(path):
    matlab_abspath='/usr/local/bin/matlab'
    savepath = '/'.join(path.split('/')[:-1]+['COMPS/'])
    save_fname = path.split('/')[-1].split('.')[0]+'_compounds.mat'
    
    processed = next(os.walk(savepath))[-1]
    command = [matlab_abspath,'-nodisplay','-nodesktop','-r', f'"get_outlier_wm_csf {path} {savepath} ;exit"']
    if save_fname not in processed:
        error_occur = os.system(' '.join(command))
        _ = os.remove(path.replace('.gz',''))
        if error_occur:
            print(path)

hcp_orig_dataset_root = '/media/das/Human_Connectome_Project/'
hcp_data = []

for dirpath, _, filenames in os.walk(hcp_orig_dataset_root):
    comp_dir = os.path.join(dirpath,'COMPS')
    if not os.path.isdir(comp_dir):
        os.mkdir(comp_dir)
        
    for fname in filenames:
        fullpath = os.path.join(dirpath,fname)
        if 'nii.gz' in fullpath:
            hcp_data.append(fullpath)
            
hcp_data = sorted(hcp_data)

Pool(12).map(extract_comp, hcp_data)
'''
pbar = tqdm(total=len(hcp_data))
pbar.set_description('Exctract Compounds : ')

def update(result):
    pbar.update(1)

for path in hcp_data:
    mp.apply_async(extract_comp, args=(path,), callback=update)
    
mp.close()
mp.join()
pbar.close()
'''