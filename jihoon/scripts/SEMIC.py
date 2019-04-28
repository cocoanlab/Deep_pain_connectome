import pandas as pd
import numpy as np
import os

from multiprocessing import Pool
from tqdm import tqdm

class load_dataset:
    def __init__(self, dataset_path='/data/SEMIC/', workers=4, shuffle=False, split_ratio=0.8, 
                 batch_size = 20, subj_group_size=2):
        self.workers = workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subj_group_size = subj_group_size
        
        labels = []
        semic_pain_path = []
        semic_rest_path = []

        self.__label_list = ['model02_Overall_FIR_SPM_SINGLE_TRIAL', 'model02_FIR_SPM_SINGLE_TRIAL']
        self.__label_names = ['rest', 'pain']
        self.__phase_list = ['train', 'valid']

        self.semic_path = {name : {} for name in self.__label_names}
        
        for dirpath, _, filenames in os.walk(dataset_path):
            for subj_num in dirpath.split('/'):
                if 'sub-semic' in subj_num:
                    subj_num = ''.join([l for l in subj_num if ord(l) >= 48 and ord(l) <= 57])

                    if subj_num != '017': # subject 17 has a problem
                        try : 
                            for name in self.__label_names : 
                                self.semic_path[name][subj_num]
                        except : 
                            for name in self.__label_names : 
                                self.semic_path[name][subj_num] = []
                        for path in filenames:
                            path = '/'.join([dirpath, path])
                            if 'beta' in path and self.__label_list[0] in path :
                                self.semic_path[self.__label_names[0]][subj_num].append(path)
                            elif 'beta' in path and self.__label_list[1] in path :
                                self.semic_path[self.__label_names[1]][subj_num].append(path)
                    break

        subj_list = list(self.semic_path[self.__label_names[0]])
        num_valid = len(subj_list) - int(len(subj_list) * split_ratio)
        
        self.subj_list = {}
        self.subj_list['valid'] = list(np.random.choice(subj_list, num_valid))
        self.subj_list['train'] = [num for num in subj_list if num not in self.subj_list['valid']]

        self.__semic_sampled_idx = list(np.random.choice(
            range(len(self.semic_path[self.__label_names[1]][self.subj_list['train'][0]])),
            len(self.semic_path[self.__label_names[0]][self.subj_list['train'][0]]),
            replace=False))
        
        self.num_subj_group = {phase : 0 for phase in self.__phase_list}
        self.__subj_batch_list = {phase : [] for phase in self.__phase_list}
        
        for phase in self.__phase_list:
            
            self.num_subj_group[phase] = int(len(self.subj_list[phase])/subj_group_size)

            if len(self.subj_list[phase]) % self.subj_group_size != 0:
                self.num_subj_group[phase] += 1

            for n in range(self.num_subj_group[phase]):
                start = n*(self.subj_group_size)
                end = (n+1)*(self.subj_group_size)
                start, end = map(int,[start,end])
                if len(self.subj_list[phase]) % self.subj_group_size != 0 :
                    if n != self.num_subj_group[phase]-1:
                        self.__subj_batch_list[phase].append(self.subj_list[phase][start:end])
                    else :
                        self.__subj_batch_list[phase].append(self.subj_list[phase][start:])
                elif len(self.subj_list[phase]) % self.subj_group_size == 0 :
                    self.__subj_batch_list[phase].append(self.subj_list[phase][start:end])
        
        
        print('Train dataset Subject list : ' + ', '.join(self.subj_list['train']) + '\n')
        print('Validation dataset Subject list : ' + ', '.join(self.subj_list['valid']) + '\n')

    def read_npy(self, path):
        for label in path.split('/'):
            if label in self.__label_list:
                label = self.__label_list.index(label)
                break

        file_num = path.split('/')[-1]
        file_num = ''.join([l for l in file_num if ord(l) >= 48 and ord(l) <= 57])
        file_num = int(file_num)-1

        if label == 1 and file_num not in self.__semic_sampled_idx : 
            return None, None
        else : 
            return np.load(path), label
            
    def load(self, phase):
        if phase not in self.__phase_list : raise ValueError('phase must be "train" or "valid".')
        phase = phase.lower()
            
        try : 
            self.current_subj_batch
            del self.batchset
        except : 
            self.current_subj_batch = {phase : 0 for phase in self.__phase_list}
        
        current_path_list = {name : [] for name in self.__label_names}
        
        for name in self.__label_names:
            for num in self.__subj_batch_list[phase][self.current_subj_batch[phase]]:
                current_path_list[name] += self.semic_path[name][num]
        
        loaded_dataset = {
            name : Pool(processes=self.workers).map(self.read_npy, current_path_list[name])
            for name in self.__label_names
        }

        for name in self.__label_names:
            fmri_data = []
            fmri_label = []
            
            for fmri, label in loaded_dataset[name]:
                if label != None:
                    fmri_data.append(fmri[np.newaxis])
                    fmri_label.append([label])
                    
            loaded_dataset[name] = {}
            loaded_dataset[name]['fmri'] = np.concatenate(fmri_data, axis=0)[:,:,:,:,np.newaxis]
            loaded_dataset[name]['labels'] = np.concatenate(fmri_label, axis=0)

        if self.current_subj_batch[phase]+1 == self.num_subj_group[phase]:
            self.current_subj_batch[phase] = 0
        else :
            self.current_subj_batch[phase]+=1

        if self.shuffle :
            indices = np.random.permutation(len(loaded_dataset[self.__label_names[-1]]['labels']))
            for name in self.__label_names:
                loaded_dataset[name]['fmri'] = loaded_dataset[name]['fmri'][indices]
                loaded_dataset[name]['labels'] = loaded_dataset[name]['labels'][indices]
            
        self.num_batchs = int(len(fmri_label)/self.batch_size)
        if len(fmri_label) % self.batch_size != 0:
            self.num_batchs+=1

        self.batchset = {name : [] for name in ['fmri', 'labels']}

        for n in range(self.num_batchs):
            start = n*(self.batch_size/2)
            end = (n+1)*(self.batch_size/2)
            start, end = map(int,[start,end])
            if len(fmri_label) % self.batch_size !=0:
                fmri = []
                label = []
                for name in self.__label_names:
                    if n != self.num_batchs-1:
                        fmri.append(loaded_dataset[name]['fmri'][start:end])
                        label.append(loaded_dataset[name]['labels'][start:end])
                    else : 
                        fmri.append(loaded_dataset[name]['fmri'][start:])
                        label.append(loaded_dataset[name]['labels'][start:])
                
            elif len(fmri_label) % self.batch_size ==0:
                fmri = []
                label = []
                for name in self.__label_names:
                    fmri.append(loaded_dataset[name]['fmri'][start:end])
                    label.append(loaded_dataset[name]['labels'][start:end])
                    
            self.batchset['fmri'].append(np.concatenate(fmri, axis=0))
            self.batchset['labels'].append(np.concatenate(label, axis=0))
            
            indices = np.random.permutation(len(self.batchset['labels'][-1]))
            for name in ['fmri', 'labels']:
                self.batchset[name][-1] = self.batchset[name][-1][indices]

        del loaded_dataset