import os
import numpy as np
import nibabel as nib

from tqdm import tqdm
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask, unmask
from multiprocessing import Pool

def normalize(raw_path, mask_path, mode='spatial'):
    raw = nib.load(raw_path)
    mask = nib.load(mask_path)
    
    masker = NiftiMasker(mask_strategy='epi')
    fitted_masker = masker.fit(mask)
    masker_img = fitted_masker.mask_img_
    
    output_path = raw_path.replace('Human_Connectome_Project', 'prep_HCP')
    non_zero_voxels = apply_mask(raw, masker_img)
    
    if mode == 'spatial':
        normalized = []

        for tr in non_zero_voxels:
            tr = tr.reshape(-1)
            tr -= tr.mean()
            tr /= tr.std()
            normalized.append(tr[np.newaxis])

        normalized = np.concatenate(normalized, axis=0)
        
    elif mode == 'temporal':
        non_zero_voxel_means = []
        non_zero_voxel_std = []
        ebs = 7./3 - 4./3 -1

        for voxels in non_zero_voxels.transpose(1,0):
            m = voxels.mean()
            std = voxels.std()
            std = std if std != 0 else ebs

            non_zero_voxel_means.append(m)
            non_zero_voxel_std.append(std)

        non_zero_voxel_means = np.array(non_zero_voxel_means)
        non_zero_voxel_std = np.array(non_zero_voxel_std)
        normalized = np.array([(v - non_zero_voxel_means)/non_zero_voxel_std for v in non_zero_voxels])
        
    elif mode == 'subjectwise':
        orig_shape = non_zero_voxels.shape
        non_zero_voxels = non_zero_voxels.reshape(-1)
        non_zero_voxels -= non_zero_voxels.mean()
        non_zero_voxels /= non_zero_voxels.std()
        normalized = non_zero_voxels.reshape(orig_shape)
        
    else :
        raise ValueError("normalization method should be among the followings : 'spatial', 'temporal', 'subjectwise'.")
        
        
    normalized = unmask(normalized, masker_img)
    nib.save(normalized, output_path)
    
def __update(result):
    pbar.update(1)

dataset_root = '/media/das/Human_Connectome_Project'
output_root = '/media/das/prep_HCP'

hcp_data_path = []

for dirlist, _, filelist in os.walk(dataset_root):
    for fname in filelist:
        hcp_data_path.append('/'.join([dirlist,fname]))
    
uniq_subj_num = list(set([path.split('/')[-2] for path in hcp_data_path]))

for subj in uniq_subj_num:
    new_path = output_root+'/'+subj
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        
multiprocessor = Pool(processes=30)
pbar = tqdm(total=len(hcp_data_path))
    
for path in hcp_data_path: 
    multiprocessor.apply_async(normalize_temporal, args=(path, './gray_matter_mask.nii', 'subjectwise'), callback=__update)
    
multiprocessor.close()
multiprocessor.join()
pbar.close()
