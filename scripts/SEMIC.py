""" 
[ Python runtime version : 3.5 ]

Title :
    SEMIC dataset Pipeline

description :
    Will be added when this function get fixed as official in this project.

arguments :
    dataset_path (No Default) :
        Insert dataset path. Dataset directory should be the following structure : 
        [D] : Directory 
        [F] : File
        #--------------------------------------#
        [D] /path/to/SEMIC/
            ┕[D] model02_FIR_SPM_SINGLE_TRIAL
                ┕[D] sub-semic001
                    ┕[F] beta_0001.npy
                    ┕[F] beta_0002.npy
                    ┕[F] ...
                    ┕[F] beta_6441.npy
                ┕[D] sub-semic002
                    ┕[F] beta_0001.npy
                    ┕[F] beta_0002.npy
                    ┕[F] ...
                    ┕[F] beta_6441.npy
                ┕[D] ...
                ┕[D] sub-semic059
            ┕[D]model02_Overall_FIR_SPM_SINGLE_TRIAL/ 
        #--------------------------------------#
        Dataset should be consist of npy files only. (Recommanded)
        
    shuffle : False (Default) :
        set whether make dataset shuffle or not
        
    split_ratio : 0.8 (Default) :
        ratio for split dataset as train and test
        
    batch_size : 20 (Default) :
        set batch size for input tensor
        
        
"""

import pandas as pd
import numpy as np
import os, gc, copy

class load_dataset:
    def __init__(self, dataset_path='/data/SEMIC/', shuffle=False, split_ratio=0.9, batch_size = 20, nan_to_zero=False):
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_ratio = split_ratio
        self.nan_to_zero = nan_to_zero
        
        labels = []
        semic_pain_path = []

        self.__label_list = ['model02_Overall_FIR_SPM_SINGLE_TRIAL', 'model02_FIR_SPM_SINGLE_TRIAL']
        self.__label_names = ['rest', 'pain']
        self.__phase_list = ['train', 'valid', 'test']
        self.__data_type = ['fmri', 'labels']
        self.__is_validset_full = False

        # save path seperately the following condition : labels, subject number
        self.semic_path = {name : {} for name in self.__label_names}
        
        # Get dataset absolute path for each npy files.
        for dirpath, _, filenames in os.walk(dataset_path):
            for subj_num in dirpath.split('/'):
                if 'sub-semic' in subj_num:
                    # Extract Subject number from path
                    subj_num = ''.join([l for l in subj_num if ord(l) >= 48 and ord(l) <= 57])

                    if subj_num != '017': # subject 17 has a problem
                        # check 'semic_path' has keys that are indicated by subject numbers
                        # if not, create a list with subject number as key
                        try :
                            for name in self.__label_names : 
                                self.semic_path[name][subj_num]
                        except : 
                            for name in self.__label_names : 
                                self.semic_path[name][subj_num] = []
                        
                        # Insert absolute path considering labels in each corresponding subject number 
                        # semic_path 
                        # ┕['001', '002', ...]
                        #  ┕['rest', 'pain']
                        for path in filenames:
                            path = '/'.join([dirpath, path])
                            
                            # rest (non-pain) state 
                            if 'beta' in path and self.__label_list[0] in path :
                                self.semic_path[self.__label_names[0]][subj_num].append(path)
                            # pain state 
                            elif 'beta' in path and self.__label_list[1] in path :
                                self.semic_path[self.__label_names[1]][subj_num].append(path)
                    break

        # Split dataset with subject numbers by given split_ratio
        exist_subj_list = list(self.semic_path[self.__label_names[0]])
        num_test = len(exist_subj_list) - int(len(exist_subj_list) * split_ratio)
        
        # each subject list takes given unique subject numbers
        self.subj_list = {}
        
        # pick subject number randomly
        self.subj_list['test'] = list(np.random.choice(exist_subj_list, num_test, replace=False))
        # takes subject numbers that is not existed in test subject number list
        for phase in self.__phase_list[:-1]:
            self.subj_list[phase] = [num for num in exist_subj_list if num not in self.subj_list['test']]

        # get about 2200 numbers in range from 0 to max number of pain case
        # rest state and pain state are in unbalance with number of dataset
        # This sampled indices take care of this
        sampled_idx = list(np.random.choice(
            range(len(self.semic_path[self.__label_names[1]][self.subj_list['train'][0]])),
            len(self.semic_path[self.__label_names[0]][self.subj_list['train'][0]]),
            replace=False))
        
        # make dataset balanced between pain and rest.
        for subj_num in self.semic_path['pain'].keys():
            self.semic_path['pain'][subj_num] = list(np.array(self.semic_path['pain'][subj_num])[sampled_idx])
            
        # pick number of beta for validation dataset.
        valid_sampled_idx = list(np.random.choice(
            sampled_idx, int(len(sampled_idx)*(1-self.split_ratio)), replace=False))
            
        # Split dataset as train and valid.
        dataset_path = {phase : copy.deepcopy(self.semic_path) for phase in self.__phase_list}

        ## Train and Validation Dataset Paths
        for label in self.__label_names:
            for subj_num in dataset_path['train'][label].keys():
                train_path = []
                valid_path = []
                for idx, path in enumerate(dataset_path['train'][label][subj_num]):
                    if idx not in valid_sampled_idx:
                        train_path.append(path)
                    elif idx in valid_sampled_idx:
                        valid_path.append(path)
                
                dataset_path['train'][label][subj_num] = train_path
                dataset_path['valid'][label][subj_num] = valid_path

        # filter test subjects in train and validation dataset, and configure testset
        for label in self.__label_names:
            for phase in self.__phase_list[:-1]:
                for subj_num in self.subj_list['test']:
                    dataset_path[phase][label].pop(subj_num)
            for subj_num in self.subj_list['train']:
                dataset_path['test'][label].pop(subj_num)
        
        # make dataset to batchset
        self.batch_pathset = {}
        
        path_list = {}
        for phase in self.__phase_list:
            path_list[phase] = {}
            for label in self.__label_names:
                path_list[phase][label] = []
                for subj_num in self.subj_list[phase]:
                    path_list[phase][label]+=dataset_path[phase][label][subj_num]
                    
                num_batchs = int(len(path_list[phase][label])/self.batch_size)
                if len(path_list[phase][label]) % self.batch_size != 0:
                    num_batchs+=1 
                    
                path_list[phase][label] = np.array(path_list[phase][label])
                
                if self.shuffle :
                    indices = np.random.permutation(len(path_list[phase][label]))
                    path_list[phase][label] = path_list[phase][label][indices]
                    
                path_list[phase][label] = np.array_split(path_list[phase][label], num_batchs*2, axis=0)
                
            self.batch_pathset[phase] = []
            for i in range(len(path_list[phase][label])):
                batch = []
                for label in self.__label_names:
                    batch += list(path_list[phase][label][i])
                self.batch_pathset[phase].append(batch)
        
        # Print which phase list subjects numbers belong.
        print('Train dataset Subject list : ' + ', '.join(self.subj_list['train']) + '\n')
        print('Test dataset Subject list : ' + ', '.join(self.subj_list['test']) + '\n')
        
        # delete unneeded variables.
        del exist_subj_list, num_test, sampled_idx, valid_sampled_idx, dataset_path, path_list
        
    def read_npy(self, path, nan_to_zero=False):
        """
        description :
            Load brain data

        arguments :
            path (npy file) / (No Default) :
                Insert dataset path to get dataset from npy file

        output :
            ( numpy array ) - (79, 95, 79) SEMIC Brain data
            ( integer ) - 0 for rest(non-pain), 1 for pain
        """
        
        # Extract label number from given path
        for label in path.split('/'):
            if label in self.__label_list:
                label = self.__label_list.index(label)
                break
        
        data = np.load(path)
        data = np.nan_to_num(data)
        
        return data, label
    
    def load(self, phase):
        """
        description :
            Load brain data with creating labels, and make dataset as batch.
        arguments :
            phase ('train' or 'valid') / (No Default) :
                Insert current working phase to read corresponding dataset.
        output :
            ( list ) - [number of batch]:(batch_size, 79, 95, 79, 1) SEMIC Brain data
                self.batchset['fmri'][???]
            ( integer ) - 0 for rest(non-pain), 1 for pain
                self.batchset['labels'][???]
        """
        # Error Exception
        if phase not in self.__phase_list : raise ValueError('phase must be "train", "valid" or "test".')
        phase = phase.lower()
        
        try : 
            self.current_subj_batch
            del self.fmri, self.labels
        except : 
            self.current_subj_batch = {phase : 0 for phase in self.__phase_list}
            
        current_dataset = self.batch_pathset[phase][self.current_subj_batch[phase]]
        self.fmri = []
        self.labels = []
        
        for path in current_dataset:
            fmri, labels = self.read_npy(path, self.nan_to_zero)
            self.fmri.append(fmri[np.newaxis])
            self.labels.append(labels)
            
        self.fmri = np.concatenate(self.fmri)
        self.labels = np.array(self.labels)
            
        if self.shuffle:
            indices = np.random.permutation(len(self.labels))
            self.fmri = self.fmri[indices]
            self.labels = self.labels[indices]
        
        if self.current_subj_batch[phase]+1 == len(self.batch_pathset[phase]):
            self.current_subj_batch[phase] = 0
        else :
            self.current_subj_batch[phase]+=1