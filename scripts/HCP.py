import warnings ; warnings.filterwarnings('ignore')
import os, gc
import numpy as np
import nibabel as nib

from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask, unmask
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.model_selection import ShuffleSplit

class HCP:
    def __init__(self, dataset_path, batch_size = 50, load_size = 300, gray_matter_only=False, mask_path=None,
                 shuffle=False, workers=4, num_total_K=5, test_K = 1):
        
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
        
        Kfold_idx = ShuffleSplit(n_splits=num_total_K, test_size=1/num_total_K).split(self.hcp_path)
        Kfold_idx = [kfold for kfold in Kfold_idx]
        
        self.path_batchset = {}
        self.path_len = {}
        self.batchset = {}
        self.current_idx = {phase : 0 for phase in self.phase_list}
        
        self.path_batchset['train'] = [self.hcp_path[idx] for idx in Kfold_idx[test_K][0]]
        self.path_batchset['valid'] = [self.hcp_path[idx] for idx in Kfold_idx[test_K][1]]
        
        for phase in self.phase_list:
            path_length = len(self.path_batchset[phase])
            if path_length % load_size != 0 :
                load_len = int(path_length/load_size)+1
            else :
                load_len = int(path_length/batch_size)
            
            split_load = []
            
            for n in range(load_len):
                if path_length % load_size !=0:
                    start = n*load_size
                    end = (n+1)*load_size
                    split_load.append(self.path_batchset[phase][start:end] if n != load_len-1 else self.path_batchset[phase][start:])
                elif load_len % load_size ==0:
                    start = n*load_size
                    end = (n+1)*load_size
                    split_load.append(self.path_batchset[phase][start:end])
            self.path_batchset[phase] = split_load
        
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
        self.batchset.clear()
        idx = self.current_idx[phase]
        path_list = self.path_batchset[phase][idx]

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
        if not self.gray_matter_only :
            self.batchset['fmri'] = self.batchset['fmri'][:,:,:,:,np.newaxis]
        
        if self.batch_size == None : pass
        else :
            data_len = len(self.batchset['fmri'])
            self.num_batchs = int(data_len/self.batch_size)+1 if data_len % self.batch_size !=0 else int(data_len/self.batch_size)
            
            fmri_batchset = []
            task_batchset = []
            
            for n in range(self.num_batchs):
                if data_len % self.batch_size !=0:
                    start = n*self.batch_size
                    end = (n+1)*self.batch_size
                    
                    fmri_batchset.append(self.batchset['fmri'][start:end] if n != self.num_batchs-1 else self.batchset['fmri'][start:])
                    task_batchset.append(self.batchset['task'][start:end] if n != self.num_batchs-1 else self.batchset['task'][start:])
                elif data_len % self.batch_size ==0:
                    start = n*self.batch_size
                    end = (n+1)*self.batch_size
                    
                    fmri_batchset.append(self.batchset['fmri'][start:end])
                    task_batchset.append(self.batchset['task'][start:end])

            self.batchset.clear()
            self.batchset = {'fmri':fmri_batchset, 'task':task_batchset}
        
        if idx == len(self.path_batchset[phase])-1:
            self.is_last = True
            self.current_idx[phase] = 0
        else :
            self.is_last = False
            self.current_idx[phase] += 1
            
        del fmri, labels
        gc.collect()