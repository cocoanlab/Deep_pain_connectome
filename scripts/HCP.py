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
                 shuffle=False, workers=4, num_total_K=5, test_K = 1, split_2d_at=None, SEED=201703):
        
        if gray_matter_only and split_2d_at is not None:
            raise ValueError('You cannot split gray matter data(1D) as 2D data.')
        
        self.task_list = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
        self.phase_list = ['train', 'valid']
        
        self.batch_size = batch_size
        self.load_size = load_size
        self.gray_matter_only = gray_matter_only
        self.workers = workers
        self.shuffle = shuffle
        self.split_2d_at = split_2d_at
        
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
        
        Kfold_idx = ShuffleSplit(n_splits=num_total_K, test_size=1/num_total_K, random_state=SEED).split(self.hcp_path)
        Kfold_idx = [kfold for kfold in Kfold_idx]
        self.Kfold_idx = Kfold_idx
        
        self.path_batchset = {}
        self.path_len = {}
        self.batchset = {}
        self.current_idx = {phase : 0 for phase in self.phase_list}
        
        self.path_batchset['train'] = [self.hcp_path[idx] for idx in Kfold_idx[test_K][0]]
        self.path_batchset['valid'] = [self.hcp_path[idx] for idx in Kfold_idx[test_K][1]]
        
        for phase in self.phase_list : 
            data_length = len(self.path_batchset[phase])
            rest = int(data_length % self.load_size)
            num_batchs = int(data_length/self.load_size)
            
            if rest == 0:
                self.path_batchset[phase] = np.array_split(self.path_batchset[phase], num_batchs)
            elif rest != 0:
                rest_batch = self.path_batchset[phase][-rest:]
                self.path_batchset[phase] = np.array_split(self.path_batchset[phase][:-rest], num_batchs)
                self.path_batchset[phase].append(rest_batch)
            
    def load_data(self, path):
        raw = nib.load(path)
        task = path.split('/')[-1].split('_')[1]
        task = self.task_list.index(task)
        task = [task for _ in range(raw.get_shape()[-1])]
        if self.gray_matter_only :
            data = apply_mask(raw, self.__masker_img)
        else :
            data = raw.get_data()
            if self.split_2d_at is None :
                data = data.transpose(3,0,1,2)
            elif self.split_2d_at == 'sagittal' :  
                data = data.transpose(3,0,1,2)
                _,_,a,b = data.shape
                data = data.reshape(-1,a,b)
            elif self.split_2d_at == 'coronal' :  
                data = data.transpose(3,1,0,2)
                _,_,a,b = data.shape
                data = data.reshape(-1,a,b)
            elif self.split_2d_at == 'axial' :  
                data = data.transpose(3,2,0,1)
                _,_,a,b = data.shape
                data = data.reshape(-1,a,b)
            else :
                raise ValueError('Brain data plains must be one of among "sagittal", "coronal", and "axial".')
            
        task = path.split('/')[-1].split('_')[1]
        task = self.task_list.index(task)
        task = [task for _ in range(data.shape[0])]
        if self.shuffle :
            indices=np.random.permutation(len(data))
            data = data[indices]
            task = [task[i] for i in indices]
        
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
            if self.split_2d_at is None :
                self.batchset['fmri'] = self.batchset['fmri'][:,:,:,:,np.newaxis]
            else :
                self.batchset['fmri'] = self.batchset['fmri'][:,:,:,np.newaxis]
        
        if self.batch_size == None : pass
        else :
            data_length = len(self.batchset['fmri'])
            rest = int(data_length % self.batch_size)
            num_batchs = int(data_length/self.batch_size)
            
            if rest == 0:
                self.batchset['fmri'] = np.array_split(self.batchset['fmri'], num_batchs)
                self.batchset['task'] = np.array_split(self.batchset['task'], num_batchs)
            elif rest != 0:
                rest_fmri_batch = self.batchset['fmri'][-rest:]
                self.batchset['fmri'] = np.array_split(self.batchset['fmri'][:-rest], num_batchs)
                self.batchset['fmri'].append(rest_fmri_batch)
                
                rest_task_batch = self.batchset['task'][-rest:]
                self.batchset['task'] = np.array_split(self.batchset['task'][:-rest], num_batchs)
                self.batchset['task'].append(rest_task_batch)
        
        if idx == len(self.path_batchset[phase])-1:
            self.is_last = True
            self.current_idx[phase] = 0
        else :
            self.is_last = False
            self.current_idx[phase] += 1
            
        del fmri, labels
        gc.collect()