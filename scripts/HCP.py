import warnings ; warnings.filterwarnings('ignore')
import gc, pickle
import numpy as np
import nibabel as nib
import tensorflow as tf

from nilearn.image import resample_img
from multiprocessing import Pool
from tqdm import tqdm

class HCP:
    def __init__(self, HCP_dataset_path, batch_size, N_fold=10, seed=201703,
                 template_nii=None, additional_task=None, additional_path=None):
        if template_nii is not None:
            self.template = nib.load(template_nii)
        if additional_task is not None and type(additional_task) != list :
            raise TypeError('additional_task should be given as list.')
        if additional_path is not None and type(additional_path) != list :
            raise TypeError('additional_path should be given as list.')
        
        np.random.seed(seed)
        np.random.shuffle(HCP_dataset_path)
        
        self.batch_size = batch_size
        self.Kfold_path = np.array_split(HCP_dataset_path,N_fold)
        if additional_path is not None and additional_task is not None:
            additional_Kfold_path = np.array_split(additional_path,N_fold)
            self.Kfold_path = [np.concatenate((HCP, add)) for HCP, add in zip(self.Kfold_path, additional_Kfold_path)]
        
        self.batchset = {}
        
        __dump_dat = pickle.load(open(self.Kfold_path[0][0],'rb'))
        self.main_tasks = sorted(set([task.split('_')[0] for task in __dump_dat.keys()]))
        self.sub_tasks = sorted(set([subtask for task in __dump_dat.values() for subtask in list(task.keys())]))
        if additional_task is not None :
            self.main_tasks += additional_task
            self.sub_tasks += additional_task
            
        self.total_sample_num = 0
        self.total_batch_num = 0
    
    def _count_num_data(self,path):
        with open(path,'rb') as fin:
            total_len = sum([1 for dat in pickle.load(fin).values() for _ in dat.values()])
            
        idx_start = list(range(0,total_len,self.batch_size))[:-1]
        idx_end = list(range(0,total_len,self.batch_size))[1:]
        cut_s_e = [[s,e] for s,e in zip(idx_start,idx_end)]
        if idx_end[-1] < total_len:
            cut_s_e.append([idx_end[-1],total_len])
        return total_len, len(cut_s_e)
            
    def count_num_batch(self, validset_idx=None, n_jobs=36):
        if  validset_idx is not None :
            train_path = [self.Kfold_path[i] for i in range(len(self.Kfold_path)) if i != validset_idx]
            train_path = np.concatenate(train_path)
        else :
            train_path = np.concatenate(self.Kfold_path)
        mp = Pool(processes=n_jobs)
        print('counting number of training batchset ...')
        
        total_sample_cnt = []
        total_batch_cnt = []
        pbar = tqdm(total=len(train_path))
        def update(result):
            total_sample_cnt.append(result[0])
            total_batch_cnt.append(result[1])
            pbar.update(1)
        
        
        for path in train_path:
            mp.apply_async(self._count_num_data, args=(path,), callback=update)

        mp.close()
        mp.join()
        self.total_sample_num = sum(total_sample_cnt)
        self.total_batch_num = sum(total_batch_cnt)
        print(f'total number of samples = {self.total_sample_num}')
        print(f'total number of batch = {self.total_batch_num}')
        
    def load(self, path):
        img_list = []
        lbl_list = []
        
        with open(path,'rb') as fin:
            for dat in pickle.load(fin).values():
                for task, fmri in dat.items():
                    if 'run' in task : task = 'heat_pain'
                    fmri = resample_img(fmri, self.template.get_affine(), self.template.shape)
                    img_list.append(fmri.get_fdata()[np.newaxis])
                    lbl_list.append(self.sub_tasks.index(task))
                    if fmri.in_memory : 
                        fmri.uncache()
                        del fmri
                        gc.collect()
                        
        return np.vstack(img_list), np.array(lbl_list)
        
    def load_data(self, k_idx, shuffle=False, n_jobs=36):
        self.batchset.clear()
        train_path = self.Kfold_path[k_idx]
        if shuffle :
            np.random.seed(None)
            np.random.shuffle(train_path)
        fmri = []
        task = []
        
        mp = Pool(processes=n_jobs)
        pbar = tqdm(total=len(train_path))
        
        def update(result):
            fmri.append(result[0])
            task.append(result[1])
            pbar.update(1)

        for path in train_path:
            mp.apply_async(self.load, args=(path,), callback=update)

        mp.close()
        mp.join()
        pbar.close()
        
        self.batchset['fmri'] = np.vstack(fmri)
        self.batchset['task'] = np.concatenate(task)
        fmri.clear()
        task.clear()
        del fmri,task
        gc.collect()