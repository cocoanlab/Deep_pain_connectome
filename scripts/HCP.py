import warnings ; warnings.filterwarnings('ignore')
import os, gc
import numpy as np
import nibabel as nib

from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask, unmask
from multiprocessing import Pool
from tqdm import tqdm

class HCP:
    def __init__(self, dataset_path, batch_size = 50, load_size = 300, gray_matter_only=False, mask_path=None,
                 shuffle=False, workers=4, split_mode = 'random', split_ratio=0.8, num_total_K=5, test_K = '1'):
        
        if split_mode not in ['random', 'kfold']:
            raise ValueError("split_mode should be given as None or 'Kfold:K-fold classification'.")
        
        self.task_list = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
        self.phase_list = ['train', 'valid']
        
        self.batch_size = batch_size
        self.load_size = load_size
        self.gray_matter_only = gray_matter_only
        self.workers = workers
        
        self.hcp_path = []
        self.split_size = {}

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
        elif not gray_matter_only:
            pass
        else : 
            raise ValueError('Gray matter mask is required to extract it')
        
        threshold = round(len(self.hcp_path)*0.8)
        self.train_path = self.hcp_path[:threshold]
        self.split_size['train'] = int(len(self.train_path)/self.load_size)+1
        self.valid_path = self.hcp_path[threshold:]
        self.split_size['valid'] = int(len(self.valid_path)/self.load_size)+1
        
        self.current_idx = 0
        
    def load_data(self, path):
        raw = nib.load(path)
        task = path.split('/')[-1].split('_')[1]
        task = self.task_list.index(task)
        task = [task for _ in range(raw.get_shape()[-1])]
        if self.gray_matter_only :
            data = apply_mask(raw, self.__masker_img)
        else :
            data = raw.get_data()
            data = data.transpose(3,0,1,2)
            
        if raw.in_memory : raw.uncache()
        del raw
        gc.collect()
        return data, task
    
    def load_batchset(self, phase):
        if self.current_idx > 0:
            self.batchset.clear()
            del self.batchset
            gc.collect()
        
        if phase == 'train':
            self.__split_size = self.split_size[phase]
            path_list = np.array_split(self.train_path, self.__split_size,)[self.current_idx]
        elif phase == 'valid':
            self.__split_size = self.split_size[phase]
            path_list = np.array_split(self.valid_path, self.__split_size)[self.current_idx]
        else :
            raise ValueError('sss')

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

        self.batchset = {}
        self.batchset['fmri'] = np.concatenate(fmri, axis=0)
        self.batchset['task'] = np.concatenate(labels, axis=0)
        fmri.clear()
        labels.clear()
        
        batch_len = round(len(self.batchset['task'])/self.batch_size)
        self.batchset['fmri'] = np.array_split(self.batchset['fmri'], batch_len)
        self.batchset['task'] = np.array_split(self.batchset['task'], batch_len)
        
        if bool(int(self.current_idx/self.__split_size)):
            self.is_last = True
            self.current_idx = 0
        else :
            self.is_last = False
            self.current_idx+=1
            
        del fmri, labels
        gc.collect()