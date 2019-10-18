import warnings ; warnings.filterwarnings('ignore')
import os, gc
import numpy as np
import nibabel as nib

from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask, unmask
from multiprocessing import Pool
from tqdm import tqdm

class HCP:
    def __init__(self, dataset_path, batch_size = 50, load_size = 300, gray_matter_only=False, mask_path=None, workers=4):
        self.task_list = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
        
        self.batch_size = batch_size
        self.load_size = load_size
        self.gray_matter_only = gray_matter_only
        self.workers = workers
        
        self.hcp_path = []

        for dirlist, _, filelist in os.walk(dataset_path):
            data_count = 0
            tmp_path = []
            for fname in filelist:
                data_count+=1
                tmp_path.append('/'.join([dirlist,fname]))
            if data_count == 14:
                self.hcp_path+=tmp_path
        
        if gray_matter_only and mask_path is not None:
            masker = NiftiMasker(mask_strategy='epi')
            fitted_masker = masker.fit(mask_path)
            self.__masker_img = fitted_masker.mask_img_
        else : 
            raise ValueError('Gray matter mask is required to extract it')
        
        self.__split_size = int(len(self.hcp_path)/self.load_size)+1
        self.current_idx = 0
        
    def load_data(self, path):
        raw = nib.load(path)
        task = path.split('/')[-1].split('_')[1]
        task = self.task_list.index(task)
        task = [task for _ in range(raw.get_shape()[-1])]
        if self.gray_matter_only :
            gray = apply_mask(raw, self.__masker_img)
            if raw.in_memory : raw.uncache()
            del raw
            gc.collect()
            return gray, task
        else :
            return raw, task
    
    def load_batchset(self):
        path_list = np.array_split(self.hcp_path, self.__split_size)[self.current_idx]

        multiprocessor = Pool(processes=self.workers)
        pbar = tqdm(path_list)
        fmri=[]
        labels=[]

        def __update(result) :
            data, task = result
            fmri.append(data)
            labels.append(task)
            pbar.update(1)

        for path in path_list:
            multiprocessor.apply_async(self.load_data, args=(path,), callback=__update)

        multiprocessor.close()
        multiprocessor.join()
        pbar.close()

        batch_len = round(len(fmri)/self.load_size)
        task = np.concatenate(labels)
        self.batchset = {}
        self.batchset['fmri'] = np.array_split(fmri, batch_len)
        self.batchset['task'] = np.array_split(task, batch_len)
        
        fmri.clear()
        labels.clear()
        
        if bool(int(self.current_idx/self.__split_size)):
            self.is_last = True
            self.current_idx = 0
        else :
            self.is_last = False
            
        del fmri, labels, 
        gc.collect()