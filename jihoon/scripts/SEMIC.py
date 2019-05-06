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
        
    subj_group_size : 2 (Default) :
        
"""

import pandas as pd
import numpy as np
import os, gc

class load_dataset:
    def __init__(self, dataset_path='/data/SEMIC/', shuffle=False, split_ratio=0.9, 
                 batch_size = 20, subj_group_size=2, nan_to_zero=False):
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subj_group_size = subj_group_size
        self.split_ratio = split_ratio
        self.nan_to_zero = nan_to_zero
        
        labels = []
        semic_pain_path = []

        self.__label_list = ['model02_Overall_FIR_SPM_SINGLE_TRIAL', 'model02_FIR_SPM_SINGLE_TRIAL']
        self.__label_names = ['rest', 'pain']
        self.__phase_list = ['train', 'test']
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
        subj_list = list(self.semic_path[self.__label_names[0]])
        num_test = len(subj_list) - int(len(subj_list) * split_ratio)
        
        # each subject list takes given unique subject numbers
        self.subj_list = {}
        
        # pick subject number randomly
        self.subj_list['test'] = list(np.random.choice(subj_list, num_test))
        # takes subject numbers that is not existed in test subject number list
        self.subj_list['train'] = [num for num in subj_list if num not in self.subj_list['test']]

        # get about 2200 numbers in range from 0 to max number of pain case
        # rest state and pain state are in unbalance with number of dataset
        # This sampled indices take care of this
        self.__semic_sampled_idx = list(np.random.choice(
            range(len(self.semic_path[self.__label_names[1]][self.subj_list['train'][0]])),
            len(self.semic_path[self.__label_names[0]][self.subj_list['train'][0]]),
            replace=False))
        
        self.__valid_sampled_idx = list(np.random.choice(
            self.__semic_sampled_idx,
            int(len(self.__semic_sampled_idx)*(1-self.split_ratio)), replace=False))
            
        # groups how many subjects are in a shot of read_npy.
        # consider your computer RAM. 
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
        
        
        # Print which phase list subjects numbers belong.
        print('Train dataset Subject list : ' + ', '.join(self.subj_list['train']) + '\n')
        print('Test dataset Subject list : ' + ', '.join(self.subj_list['test']) + '\n')

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

        file_num = path.split('/')[-1]
        file_num = ''.join([l for l in file_num if ord(l) >= 48 and ord(l) <= 57])
        file_num = int(file_num)-1
        
        # if this path not in sampled indices list for data balance, ignore to read it
        if label == 1 and file_num not in self.__semic_sampled_idx : 
            return None, None, None
        else : 
            data = np.load(path)
            data = np.nan_to_num(data)
            if file_num in self.__valid_sampled_idx:
                return data, label, True
            else :
                return data, label, False
            
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
        if phase not in self.__phase_list : raise ValueError('phase must be "train" or "test".')
        phase = phase.lower()
            
        # check existance of variable to shows current loaded subject group.
        # if not, create current subject batch index to indicate it.
        try : 
            self.current_subj_batch
            del self.batchset
        except : 
            self.current_subj_batch = {phase : 0 for phase in self.__phase_list}
        
        # collect given subject group's whole paths that they have.
        current_path_list = {name : [] for name in self.__label_names}
        
        for name in self.__label_names:
            for num in self.__subj_batch_list[phase][self.current_subj_batch[phase]]:
                current_path_list[name] += self.semic_path[name][num]
                
        # concatenate loaded fmri data in one array of 5 Dimensions :
        # (number of brain, 79, 95, 79, 1)
        
        # load dataset seperately by labels.
        loaded_dataset = {
            name : {data : [] for data in self.__data_type}
            for name in self.__label_names
        }
        try : self.__valid_dataset
        except : 
            if not self.__is_validset_full :
                self.__valid_dataset = {
                    name : {data : [] for data in self.__data_type}
                    for name in self.__label_names
                }
            else : pass
        
        for name in self.__label_names:
            for path in current_path_list[name]:
                img, label, is_valid = self.read_npy(path, self.nan_to_zero)
                if is_valid != None:
                    img = img[np.newaxis,:,:,:,np.newaxis]
                    if is_valid and not self.__is_validset_full :
                        self.__valid_dataset[name]['fmri'].append(img)
                        self.__valid_dataset[name]['labels'].append([label])
                    else :
                        loaded_dataset[name]['fmri'].append(img)
                        loaded_dataset[name]['labels'].append([label])

            for data in self.__data_type:
                loaded_dataset[name][data] = np.concatenate(loaded_dataset[name][data], axis=0)
                
                
        # Shuffle whole dataset.
        if self.shuffle :
            indices = np.random.permutation(len(loaded_dataset[self.__label_names[-1]]['labels']))
            for name in self.__label_names:
                for data in self.__data_type :
                    loaded_dataset[name][data] = loaded_dataset[name][data][indices]
            
        # split whole dataset with given number of slices in batch.
        num_batchs = int(len(loaded_dataset[name]['labels'])/self.batch_size)
        if len(loaded_dataset[name]['labels']) % self.batch_size != 0:
            num_batchs+=1 
        
        self.batchset = {
            phase : {
                name : [] for name in self.__data_type
            } for phase in self.__phase_list
        }

        for name in self.__label_names:
            for data in self.__data_type:
                loaded_dataset[name][data] = np.array_split(loaded_dataset[name][data], num_batchs*2, axis=0)
                
        shuffled_idx = []
        for data in self.__data_type:
            for idx, (rest, pain) in enumerate(zip(loaded_dataset['rest'][data], loaded_dataset['pain'][data])):
                self.batchset[phase][data].append(np.concatenate([rest, pain], axis=0))
                if data == 'fmri' : 
                    shuffled_idx.append(np.random.permutation(len(self.batchset[phase][data][-1])))
                    self.batchset[phase][data][-1] = self.batchset[phase][data][-1][shuffled_idx[-1]]
                else :
                    self.batchset[phase][data][-1] = self.batchset[phase][data][-1][shuffled_idx[idx]]

        # move current index to next group. if count is over of number of group, initialize it 0.
        if self.current_subj_batch[phase]+1 == self.num_subj_group[phase]:
            self.current_subj_batch[phase] = 0
            
            # Validation dataset Processing
            if not self.__is_validset_full :
                self.batchset['valid'] = {name : [] for name in self.__data_type}
                for name in self.__label_names:
                    for data in self.__data_type:
                        self.__valid_dataset[name][data] = np.concatenate(self.__valid_dataset[name][data], axis=0)
                    
                # Shuffle whole dataset.
                if self.shuffle :
                    indices = np.random.permutation(len(self.__valid_dataset[self.__label_names[-1]]['labels']))
                    for name in self.__label_names:
                        for data in self.__data_type :
                            self.__valid_dataset[name][data] = self.__valid_dataset[name][data][indices]
                
                num_batchs = int(len(self.__valid_dataset[name]['labels'])/self.batch_size)
                if len(self.__valid_dataset[name]['labels']) % self.batch_size != 0:
                    num_batchs+=1 
                
                for name in self.__label_names:
                    for data in self.__data_type:
                        self.__valid_dataset[name][data] = np.array_split(self.__valid_dataset[name][data],
                                                                          num_batchs*2, axis=0)

                shuffled_idx = []
                for data in self.__data_type:
                    for idx, (rest, pain) in enumerate(zip(self.__valid_dataset['rest'][data],
                                                           self.__valid_dataset['pain'][data])):
                        
                        self.batchset['valid'][data].append(np.concatenate([rest, pain], axis=0))
                        if data == 'fmri' : 
                            shuffled_idx.append(np.random.permutation(len(self.batchset['valid'][data][-1])))
                            self.batchset['valid'][data][-1] = self.batchset['valid'][data][-1][shuffled_idx[-1]]
                        else :
                            self.batchset['valid'][data][-1] = self.batchset['valid'][data][-1][shuffled_idx[idx]]
                
                self.__is_validset_full = True
                del self.__valid_dataset
            
        else :
            self.current_subj_batch[phase]+=1
        
        # to free memory
        del loaded_dataset
        gc.collect()