import numpy as np
import pandas as pd
import warnings ; warnings.filterwarnings('ignore')
import os,gc, pickle

from copy import deepcopy
from tqdm import tqdm

class load_dataset:
    def __init__(self, dataset_path='/data/Deep_Pain_Connectome/', shuffle=False, 
                 split_ratio=0.8, batch_size = 20, mode='classification', save_division_info=False):
        
        mode = mode.lower()
        if mode not in ['classification', 'regression']:
            raise ValueError("Mode should be 'classification' or 'regression'.")
            
        self.__phase_list = ['train', 'valid', 'test']
        self.__study_list = ['bmrk3','bmrk4','nsf','ie','exp','ilcp']
        self.__data_type = ['fmri', 'label' if mode == 'classification' else 'rating']
        
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
                                
        path_list = {phase : [] for phase in self.__phase_list}
        phase_division = {study:deepcopy(path_list) for study in self.__study_list}

        for study in self.__study_list:
            subj_list = list(fmri_path[study].keys())
            num_test = len(fmri_path[study]) - int(len(fmri_path[study]) * split_ratio)
            test_subjects = np.random.choice(subj_list, num_test, replace=False)
            for subj in subj_list:
                phase = 'test' if subj in test_subjects else 'train'
                phase_division[study][phase].append(subj)

        for study in self.__study_list:
            for subj_num in phase_division[study]['train']:
                num_valid = int(len(fmri_path[study][subj_num])*(1-split_ratio))
                valid_list = np.random.choice(fmri_path[study][subj_num], num_valid)
                for path in fmri_path[study][subj_num]:
                    phase = 'valid' if path in valid_list else 'train'
                    path_list[phase].append(path)

            for subj_num in phase_division[study]['test']:
                path_list['test'].append(path)
        
        if save_division_info:
            with open('DPC_subject_phase_division.pkl', 'wb') as fout:
                pickle.dump(path_list, fout)
        
        for phase in self.__phase_list:
            indices = np.random.permutation(len(path_list[phase]))
            path_list[phase] = np.array(path_list[phase])[indices]

            num_batchs = int(len(path_list[phase])/batch_size)
            if len(path_list[phase]) % batch_size != 0:
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
                        if mode == 'classification':
                            gt_batch.append(self.__study_list.index(study))
                        else :
                            gt_batch.append(beta_ratings[study][subj_num][beta_num])

                    fmri_batch = np.concatenate(fmri_batch, axis=0)[:,:,:,:,np.newaxis]
                    gt_batch = np.array(gt_batch)
                    self.batchset[phase]['fmri'].append(fmri_batch)
                    self.batchset[phase][self.__data_type[-1]].append(gt_batch)
                    pbar.update(1)
                    