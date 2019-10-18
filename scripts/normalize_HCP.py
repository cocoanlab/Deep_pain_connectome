import os, gc
import numpy as np
import nibabel as nib

from tqdm import tqdm
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask, unmask
from multiprocessing import Pool

def normalize(raw_path, mask_path, mode='spatial', do_validate=False):
    raw = nib.load(raw_path)
    mask = nib.load(mask_path)

    masker = NiftiMasker(mask_strategy='epi')
    fitted_masker = masker.fit(mask)
    masker_img = fitted_masker.mask_img_

    output_path = raw_path.replace('Human_Connectome_Project', 'prep_HCP')
    non_zero_voxels = apply_mask(raw, masker_img)
    something_wrong = [] # For validation of normalization.

    if mode == 'spatial':
        if do_validate : 
            for idx, tr in enumerate(non_zero_voxels):
                tr = tr.reshape(-1)
                if round(tr.mean(),4) != 0 :
                    something_wrong.append(idx)
                    continue
                if round(tr.std(),4) != 1 :
                    something_wrong.append(idx)
                    continue

        elif not do_validate :
            output = []

            for tr in non_zero_voxels:
                tr = tr.reshape(-1)
                tr -= tr.mean()
                tr /= tr.std()
                output.append(tr[np.newaxis])

            normalized = np.concatenate(output, axis=0)
            output.clear()
            del output

    elif mode == 'temporal':
        if do_validate : 
            for voxels in non_zero_voxels.transpose(1,0):
                if round(voxels.mean(),4) != 0 :
                    something_wrong.append(idx)
                    continue
                if round(voxels.std(),4) != 1 :
                    something_wrong.append(idx)
                    continue

        elif not do_validate :
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
            output = [(v - non_zero_voxel_means)/non_zero_voxel_std for v in non_zero_voxels]
            normalized = np.array(output)
            output.clear()
            del non_zero_voxel_means, non_zero_voxel_std, ebs, output

    elif mode == 'subjectwise':
        if do_validate : 
            non_zero_voxels = non_zero_voxels.reshape(-1)

            if round(non_zero_voxels.means(), 4) != 0 or round(non_zero_voxels.std(), 4) != 1:
                something_wrong.append('wrong')

        elif not do_validate :
            orig_shape = non_zero_voxels.shape
            non_zero_voxels = non_zero_voxels.reshape(-1)
            non_zero_voxels -= non_zero_voxels.mean()
            non_zero_voxels /= non_zero_voxels.std()
            normalized = non_zero_voxels.reshape(orig_shape)

    else :
        raise ValueError("normalization method should be among the followings : 'spatial', 'temporal', 'subjectwise'.")

    if not do_validate :
        normalized = unmask(normalized, masker_img)
        nib.save(normalized, output_path)

    if raw.in_memory : raw.uncache()
    if mask.in_memory : mask.uncache()
    if normalized.in_memory : normalized.uncache()
    if masker_img.in_memory : masker_img.uncache()

    if do_validate :
        return (raw_path, something_wrong)
    del raw, mask, non_zero_voxels, normalized, masker, masker_img, something_wrong
    gc.collect()
    
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
        
multiprocessor = Pool(processes=27)
pbar = tqdm(total=len(hcp_data_path))
    
for path in hcp_data_path: 
    multiprocessor.apply_async(normalize, args=(path, './gray_matter_mask.nii', 'temporal'), callback=__update)
    
multiprocessor.close()
multiprocessor.join()
pbar.close()
