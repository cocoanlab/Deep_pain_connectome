from scipy.io import loadmat
from nilearn.image import resample_to_img
from multiprocessing import Pool

import nibabel as nib
import nilearn.plotting as plotting
import pandas as pd
import os

dataset_path = '/cocoanlab2/GPU1_sync/data/Deep_pain_connectome/deeppain/original_dat/img_dat/'
temp_resample = nib.load('/cocoanlab2/GPU1_sync/data/Deep_pain_connectome/HCP_1200/162329/MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR.nii.gz')
templete = temp_resample.slicer[:,:,:,0]
temp_nib = nib.load('friend_trial01.nii')
    
path_list = []

for dirpath, _, filenames in os.walk(dataset_path):
    for fname in filenames :
        path_list.append(dirpath+fname)
        
def resample(path):
    raw_data = loadmat(dirpath+fname)['recon_dat']
    raw_data = nib.Nifti1Image(raw_data, temp_nib.affine, temp_nib.header)
    
    resampled_data = []
    for i in range(int(raw_data.get_shape()[-1])):
        resampled_slice = resample_to_img(raw_data.slicer[:,:,:,i], templete)
        resampled_data.append(resampled_slice)
    resampled_data = nib.concat_images(resampled_data)
    
    path = path.replace('original_dat', 'resampled_nifti')
    path = path.replace('mat', 'nii')
    
    nib.save(resampled_data, path)
    
Pool(processes=6).map(resample, path_list)