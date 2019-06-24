import numpy as np
import pandas as pd
import warnings ; warnings.filterwarnings('ignore')
import os,gc, pickle
import datetime

from copy import deepcopy
from tqdm import tqdm

class load_dataset:
    def __init__(self, dataset_path='/data/Deep_Pain_Connectome/',
                 batch_size = 20, task='classification', split_mode = 'random', split_ratio=0.8, 
                 num_total_K=5, test_K = '1', save_division_info=False, save_batchset=False, 
                 restore=False, filename = 'DPC_subject_phase_division.pkl'):

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.task = task.lower()
        self.split_mode = split_mode.lower()
        self.split_ratio = split_ratio
        self.num_total_K = num_total_K
        self.test_K = test_K
        self.save_division_info = save_division_info
        self.save_batchset = save_batchset
        self.filename = filename
        
        if split_mode not in ['random', 'kfold']:
            raise ValueError("split_mode should be given as None or 'Kfold:K-fold classification'.")
        
        if task not in ['classification', 'regression']:
            raise ValueError("Task should be 'classification' or 'regression'.")
            
        self.__phase_list = ['train', 'valid', 'test']
        self.__study_list = ['bmrk3','bmrk4','nsf','ie','exp','ilcp']
        self.__data_type = ['fmri', 'label' if task == 'classification' else 'rating']
        
        if restore :
            with open(dataset_path, 'rb') as f:
                info = pickle.load(f)
                
            self.batch_size = info['attr']['batch_size']
            self.split_ratio = info['attr']['split_ratio']
            self.task = info['attr']['task']
            self.path_list = info['path_list']
            self.beta_ratings = info['beta_ratings']
            self.split_mode = info['attr']['split_mode']
            self.split_ratio = info['attr']['split_ratio']
            self.num_total_K = info['attr']['num_total_K']
            self.test_K = info['attr']['test_K']
            if 'batchset' in info.keys():
                self.batchset = info['batchset']
                
        else : 
            fmri_path = {study:{} for study in self.__study_list}
            beta_ratings = deepcopy(fmri_path)
            for dirpath, _, filenames in os.walk(dataset_path):
                if 'study' in dirpath :
                    for fname in filenames :
                        if 'npy' in fname or 'csv' in fname:
                            path = '/'.join([dirpath, fname])
                            study= path.split('/')[-3].split('_')[-1]

                            if 'npy' in fname:
                                subj_num = path.split('/')[-2]
                                subj_num = ''.join([l for l in subj_num if ord(l) >= 48 and ord(l) <= 57])
                                try : fmri_path[study][subj_num].append(path)
                                except : fmri_path[study][subj_num] = [path]
                            elif 'csv' in fname:
                                subj_num = path.split('/')[-1].split('_')[-2]
                                subj_num = ''.join([l for l in subj_num if ord(l) >= 48 and ord(l) <= 57])
                                rating_table = pd.read_csv(path)[['beta_index','rating_standard_bt_study']]
                                beta_ratings[study][subj_num] = {}

                                for beta, score in rating_table.get_values():
                                    beta = beta.split('.')[0]
                                    beta = ''.join([l for l in beta if ord(l) >= 48 and ord(l) <= 57])
                                    beta_ratings[study][subj_num][beta] = float(score)

            self.fmri_path = fmri_path
            self.beta_ratings = beta_ratings
            
            if self.split_mode == 'random':
                path_list = {phase : [] for phase in self.__phase_list}
                phase_division = {study:deepcopy(path_list) for study in self.__study_list}

                for study in self.__study_list:
                    subj_list = list(fmri_path[study].keys())
                    num_test = len(fmri_path[study]) - int(len(fmri_path[study]) * self.split_ratio)
                    test_subjects = np.random.choice(subj_list, num_test, replace=False)
                    for subj in subj_list:
                        phase = 'test' if subj in test_subjects else 'train'
                        phase_division[study][phase].append(subj)

                for study in self.__study_list:
                    for subj_num in phase_division[study]['train']:
                        num_valid = int(len(fmri_path[study][subj_num])*(1-self.split_ratio))
                        valid_list = np.random.choice(fmri_path[study][subj_num], num_valid)
                        for path in fmri_path[study][subj_num]:
                            phase = 'valid' if path in valid_list else 'train'
                            path_list[phase].append(path)

                    for subj_num in phase_division[study]['test']:
                        path_list['test'].append(path)
                        
            elif self.split_mode == 'kfold':
                if type(test_K) != str : test_K = str(test_K)
                    
                self.Kfoldset = {str(idx+1) : {} for idx in range(5)}

                for study in self.__study_list:
                    path_list = np.array(list(fmri_path[study].keys()))
                    indices = np.random.permutation(len(path_list))
                    path_list = path_list[indices]
                    for idx, batch in enumerate(np.array_split(path_list,num_total_K)):
                        self.Kfoldset[str(idx+1)][study] = batch
                path_list = self.change_K_fold(self.test_K, reload=False)
                
            if save_division_info:
                info = {}
                info['path_list'] = deepcopy(path_list)
                info['beta_ratings'] = self.beta_ratings
                info['attr'] = {
                    'dataset_path' : self.dataset_path,
                    'batch_size' : self.batch_size,
                    'task' : self.task,
                    'split_mode' : self.split_mode,
                    'split_ratio' : self.split_ratio,
                    'num_total_K' : self.num_total_K,
                    'test_K' : self.test_K,
                }
                if self.split_mode == 'kfold':
                    info['Kfoldset'] = self.Kfoldset
                
                now = datetime.datetime.now()
                now = '{:04d}{:02d}{:02d}-{:02d}{:02d}_'.format(now.year, now.month, now.day,
                                                                now.hour, now.minute)
                self.filename = now+self.filename
                with open(self.filename, 'wb') as fout:
                    pickle.dump(info, fout)
        try :
            self.batchset
            
        except :
            self.__load(path_list)
                    
    def change_K_fold(self, test_K, reload=False, save_batchset=False):
        self.test_K = test_K
        fold_numbers = list(self.Kfoldset.keys())
        path_list = {phase : [] for phase in self.__phase_list}
        
        for study in self.__study_list:
            for fold_num in fold_numbers:
                for subj_num in self.Kfoldset[fold_num][study]:
                    num_valid = int(len(self.fmri_path[study][subj_num])*(1-self.split_ratio))
                    valid_list = np.random.choice(self.fmri_path[study][subj_num], num_valid)
                    for path in self.fmri_path[study][subj_num]:
                        if fold_num == self.test_K :
                            path_list['test'].append(path)
                        else :                             
                            phase = 'valid' if path in valid_list else 'train'
                            path_list[phase].append(path)
        if reload :
            self.__load(path_list)
        else :
            return path_list
    
    def __load(self, path_list):
        for phase in self.__phase_list:
            indices = np.random.permutation(len(path_list[phase]))
            path_list[phase] = np.array(path_list[phase])[indices]

            num_batchs = len(path_list[phase])//self.batch_size
            if len(path_list[phase]) % self.batch_size != 0:
                num_batchs+=1 

            path_list[phase] = np.array_split(path_list[phase], num_batchs, axis=0)

        data_list = {data : [] for data in self.__data_type}
        self.batchset = {}

        for phase in self.__phase_list:
            self.batchset[phase] = deepcopy(data_list) 
            with tqdm(total=len(path_list[phase])) as pbar:
                pbar.set_description('[ {} ] load dataset '.format(phase.upper()))
                for batch_pathset in path_list[phase]:
                    fmri_batch= []
                    gt_batch = []
                    for path in batch_pathset:
                        study = path.split('/')[-3].split('_')[-1]
                        subj_num = path.split('/')[-2]
                        subj_num = ''.join([l for l in subj_num if ord(l) >= 48 and ord(l) <= 57])
                        beta_num = path.split('/')[-1].split('.')[0]
                        beta_num = ''.join([l for l in beta_num if ord(l) >= 48 and ord(l) <= 57])

                        fmri_batch.append(np.load(path)[np.newaxis])
                        if self.task == 'classification':
                            gt_batch.append(self.__study_list.index(study))
                        else :
                            gt_batch.append(self.beta_ratings[study][subj_num][beta_num])

                    fmri_batch = np.concatenate(fmri_batch, axis=0)[:,:,:,:,np.newaxis]
                    gt_batch = np.array(gt_batch)
                    self.batchset[phase]['fmri'].append(fmri_batch)
                    self.batchset[phase][self.__data_type[-1]].append(gt_batch)
                    pbar.update(1)
                    
        if self.save_batchset:
            with open(self.filename, 'rb') as f:
                info = pickle.load(f)
            info['batchset'] = self.batchset
            print('Saving Batchset ... ')
            with open(self.filename, 'wb') as fout:
                pickle.dump(info, fout)
                    