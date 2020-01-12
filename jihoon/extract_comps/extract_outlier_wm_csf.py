from multiprocessing import Pool
from tqdm import tqdm
import os

hcp_orig_dataset_root = '/media/das/Human_Connectome_Project/'
hcp_data = []

for dirpath, _, filenames in os.walk(hcp_orig_dataset_root):
    for fname in filenames:
        comp_dir = os.path.join(dirpath,'COMPS')
        if not os.path.isdir(comp_dir):
            os.mkdir(comp_dir)
        
        fullpath = os.path.join(dirpath,fname)
        if 'nii.gz' in fullpath:
            hcp_data.append(fullpath)
            
mp = Pool(20)

pbar = tqdm(total=len(hcp_data))
pbar.set_description('Exctract Compounds : ')

def extract_comp(path, matlab_abspath='/usr/local/bin/matlab'):
    savepath = '/'.join(path.split('/')[:-1]+['COMPS/'])
    command = [matlab_abspath,'-nodisplay','-nodesktop','-r', f'"get_outlier_wm_csf {path} {savepath} ;exit"']
    _ = os.system(' '.join(command))
    _ = os.remove(path.replcace('.gz',''))
    
def update(result):
    pbar.update(1)

for path in hcp_data:
    mp.apply_async(extract_comp, args=(path,), callback=update)
    
mp.close()
mp.join()
pbar.close()