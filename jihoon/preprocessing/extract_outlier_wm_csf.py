from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import os

def extract_comp(orig_path, preproc_path):
    matlab_abspath='/usr/local/bin/matlab'
    save_fname = preproc_path.split('/')[-1].split('.')[0]+'_compounds.mat'
    savepath = '/'.join(preproc_path.replace('smooth5','compounds').split('/')[:-1])+'/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    processed = next(os.walk(savepath))[-1]
    command = ['nohup',matlab_abspath,'-singleCompThread','-nodisplay','-nodesktop','-r', f'"get_outlier_wm_csf {orig_path} {preproc_path} {savepath} ;exit" > out.out']
    if save_fname not in processed:
        error_occur = os.system(' '.join(command))
        #_ = os.remove(preproc_path.replace('.gz',''))
        if error_occur:
            print(preproc_path)

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
start = 0
end = len(hcp_data)
hcp_data = [hcp_data[i] for i in sorted_order][start:end]
hcp_orig = [hcp_orig[i] for i in sorted_order][start:end]

mp = Pool(18)
pbar = tqdm(total=len(hcp_data))

def update(result):
    pbar.update(1)
    
for orig, preproc in zip(hcp_orig, hcp_data):
    mp.apply_async(extract_comp, args=(orig, preproc,), callback=update)
    
mp.close()
mp.join()
pbar.close()