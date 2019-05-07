import numpy as np
import SimpleITK as sitk
import os, pickle

from multiprocessing import Pool

def convert_nii_to_npy(os_walk):
    dirpath, dirnames, filenames = os_walk
    if 'sub-semic' in dirpath and '@eaDir' not in dirpath:
        nifti_path = [path for path in filenames if 'nii' in path and 'beta' in path]
        for path in nifti_path:
            data = sitk.ReadImage('/'.join([dirpath,path]))
            data = sitk.GetArrayFromImage(data)
            
            # save fmri data as npy with making it numpy array
            numpy_path = dirpath.split('/')
            numpy_path.insert(-2, 'npy_data')
            if not os.path.exists('/'.join(numpy_path)) : os.makedirs('/'.join(numpy_path))
            numpy_path = '/'.join(numpy_path+[path.split('.')[0]+'.npy'])
                
            np.save(numpy_path, data)

    else : pass
    
dataset_path = ["/cocoanlab2/GPU1_sync/data/SEMIC/imaging/model02_FIR_SPM_SINGLE_TRIAL/",
                "/cocoanlab2/GPU1_sync/data/SEMIC/imaging/model02_Overall_FIR_SPM_SINGLE_TRIAL/"]

for path in dataset_path:
    Pool(processes=30).map(convert_nii_data, os.walk(path))