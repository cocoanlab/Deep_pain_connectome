import os
import nibabel as nib

from multiprocessing import Pool
from nilearn.image import smooth_img, resample_img
from tqdm import tqdm


hcp_orig_dataset_root = '/media/das/Human_Connectome_Project/'
hcp_data = []
template = nib.load('./friend_trial01.nii')
'''
for dirpath, _, filenames in os.walk(hcp_orig_dataset_root):
    for fname in filenames:
        comp_dir = os.path.join(dirpath,'smooth5')
        if not os.path.isdir(comp_dir):
            os.mkdir(comp_dir)
        
        fullpath = os.path.join(dirpath,fname)
        if 'nii.gz' in fullpath:
            hcp_data.append(fullpath)
'''
for dirpath, _, filenames in os.walk(hcp_orig_dataset_root):
    for fname in filenames:
        comp_dir = os.path.join(dirpath,'smooth5')
        if not os.path.isdir(comp_dir):
            os.mkdir(comp_dir)
        
        fullpath = os.path.join(dirpath,fname)
        if 'unpro' in fullpath:
            fullpath = fullpath.replace('/unprocessed/', '/')
            fullpath = fullpath.replace('nii', 'nii.gz')
            hcp_data.append(fullpath)
            
def resampling_smoothing(fmripath, fwhm=5):
    savepath = fmripath.split('/')
    fname = savepath.pop()
    savepath = '/'.join(savepath)
    savepath = os.path.join(savepath, 'smooth5', fname.replace('nii.gz', 'nii'))
    
    data = nib.load(fmripath)
    data = resample_img(data, template.get_affine(), template.shape, force_resample=True)
    data = smooth_img(data, fwhm)
    nib.save(data, savepath)
    if data.in_memory : data.uncache()
    del data
            
mp = Pool(20)
pbar = tqdm(total=len(hcp_data))
def update(result):
    pbar.update(1)
    
for path in hcp_data:
    mp.apply_async(resampling_smoothing, args=(path,), callback=update)
    
mp.close()
mp.join()
pbar.close()