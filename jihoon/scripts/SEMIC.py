import pandas as pd
import numpy as np
import SimpleITK as sitk
import os, glob

from scipy.io import loadmat
from tqdm import tqdm

def explore_dir(dir,count=0,f_extensions=None):
    if count==0:
        global n_dir, n_file, filenames, filelocations
        n_dir=n_file=0
        filenames=list()
        filelocations=list()

    for img_path in sorted(glob.glob(os.path.join(dir,'*' if f_extensions is None else '*.'+f_extensions))):
        if os.path.isdir(img_path):
            n_dir +=1
            explore_dir(img_path,count+1)
        elif os.path.isfile(img_path):
            n_file += 1
            filelocations.append(img_path)
            filenames.append(img_path.split("/")[-1])
    return np.array((filenames,filelocations))

class load_dataset:
    def __init__(self, meta_path, dataset_path, padding=0):
        if type(padding) != int:
            raise ValueError('padding should be int.')
        
        semic_meta = loadmat(meta_path)['semic_info'][0,0][0][0]
        
        subj_score_list = []

        for subj in range(len(semic_meta)):
            if subj == 16 : continue # subj 17 data doesn't exist.
            for trial in range(len(semic_meta[subj])):
                score = float(semic_meta[subj][trial][3][0])
                file_name = semic_meta[subj][trial][4][0]
                file_name = file_name.split('/')[-1].split('.')[0]

                subj_score_list.append([subj, file_name, score])

        subj_score_list = np.array(subj_score_list)
        
        fmri_path = [{} for _ in range(len(semic_meta)-1)]

        for path in explore_dir(dataset_path)[1]:
            if subj == 16 : continue # subj 17 data doesn't exist.
            elif 'PAIN' in path and 'beta' in path:
                subj_num = int(path.split('/')[-2][-3:]) - 1
                file_name = path.split('/')[-1].split('.')[0]
                fmri_path[subj_num if subj_num < 16 else subj_num-1][file_name] = path

        img_set = []
        labels = []

        with tqdm(total=len(subj_score_list)) as pbar:
            pbar.set_description('[ load dataset ]')
            for subj, file_name, score in subj_score_list:
                subj = int(subj)
                subj = subj if subj < 16 else subj-1
                fmri_data = sitk.ReadImage(fmri_path[subj][file_name])
                fmri_data = sitk.GetArrayFromImage(fmri_data)
                img_set.append([np.nan_to_num(fmri_data)])
                labels.append(float(score))
                pbar.update(1)

        self.img_set = np.concatenate(img_set, axis=0)
        self.labels = np.array(labels)