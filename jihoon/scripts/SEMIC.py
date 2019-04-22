import pandas as pd
import numpy as np
import SimpleITK as sitk
import os, glob

from multiprocessing import Pool
from tqdm import tqdm

def explore_dir(dir,count=0,f_extensions=None):
    if count==0:
        global n_dir, n_file, filenames, filelocations
        n_dir=n_file=0
        filenames=list()
        filelocations=list()

    for img_path in sorted(glob.glob(os.path.join(dir,'*' if f_extensions is None else '*.'+f_extensions))):
        if os.path.isdir(img_path):
            n_dir +=1
            explore_dir(img_path,count+1)
        elif os.path.isfile(img_path):
            n_file += 1
            filelocations.append(img_path)
            filenames.append(img_path.split("/")[-1])
    return np.array((filenames,filelocations))

class load_dataset:
    def __init__(self, dataset_path='/data/SEMIC/', workers=4, shuffle=False, split_ratio=0.8, batch_size = 20):
        self.workers = workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        labels = []
        semic_pain_path = []
        semic_rest_path = []

        self.__label_list = ['model02_Overall_FIR_SPM_SINGLE_TRIAL', 'model02_FIR_SPM_SINGLE_TRIAL']

        self.semic_path = {name : {} for name in ['semic', 'control']}

        for path in explore_dir(dataset_path)[1]:
            for subj_num in path.split('/'):
                if 'sub-semic' in subj_num:
                    subj_num = ''.join([l for l in subj_num if ord(l) >= 48 and ord(l) <= 57])
                    break
            if subj_num != '017': # subject 17 has a problem
                try : 
                    for name in ['semic', 'control'] : 
                        self.semic_path[name][subj_num]
                except : 
                    for name in ['semic', 'control'] : 
                        self.semic_path[name][subj_num] = []

                if 'beta' in path and self.__label_list[0] in path : self.semic_path['control'][subj_num].append(path)
                elif 'beta' in path and self.__label_list[1] in path : self.semic_path['semic'][subj_num].append(path)

        subj_list = list(self.semic_path['control'])
        num_valid = len(subj_list) - int(len(subj_list) * split_ratio)

        self.valid_subj_list = list(np.random.choice(subj_list, num_valid))
        self.train_subj_list = [num for num in subj_list if num not in self.valid_subj_list]

        self.__semic_sampled_idx = list(np.random.choice(range(len(self.semic_path['semic'][self.train_subj_list[0]])),
                                                         len(self.semic_path['control'][self.train_subj_list[0]]),
                                                         replace=False))
        
        print('Train dataset Subject list : ' + ', '.join(self.train_subj_list) + '\n')
        print('Validation dataset Subject list : ' + ', '.join(self.valid_subj_list) + '\n')

    def read_nii(self, path):
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
            data = sitk.ReadImage(path)
            data = sitk.GetArrayFromImage(data)
            return data, label
            
    def load(self, phase):
        if phase not in ['train', 'valid'] : raise ValueError('phase must be "train" or "valid".')
        phase = phase.lower()
            
        try : 
            self.current_subj
            del self.batchset
        except : self.current_subj = {name : 0 for name in ['train', 'valid']}

        subj_num = self.train_subj_list[self.current_subj[phase]] if phase == 'train' else self.valid_subj_list[self.current_subj[phase]]

        loaded_dataset = {
            name : Pool(processes=self.workers).map(self.read_nii, self.semic_path[name][subj_num])
            for name in ['semic', 'control']
        }

        fmri_data = []
        fmri_label = []
        for name in ['semic', 'control']:
            for fmri, label in loaded_dataset[name]:
                if label != None:
                    fmri_data.append(fmri[np.newaxis])
                    fmri_label.append([label])

        fmri_data = np.concatenate(fmri_data, axis=0)
        fmri_label = np.concatenate(fmri_label, axis=0)

        if self.current_subj[phase]+1 == len(self.train_subj_list if phase == 'train' else self.valid_subj_list):
            self.current_subj[phase] = 0
        else :
            self.current_subj[phase]+=1

        if self.shuffle :
            indices = np.random.permutation(len(fmri_label))
            fmri_data = fmri_data[indices]
            fmri_label = fmri_label[indices]


        self.num_batchs = int(len(fmri_label)/self.batch_size)+1 if len(fmri_label) % self.batch_size !=0 else int(len(fmri_label)/self.batch_size)

        self.batchset = {name : [] for name in ['fmri', 'labels']}

        for n in range(self.num_batchs):
            start = n*self.batch_size
            end = (n+1)*self.batch_size
            if len(fmri_label) % self.batch_size !=0:
                self.batchset['fmri'].append(fmri_data[start:end] if n != self.num_batchs-1 else fmri_data[start:])
                self.batchset['labels'].append(fmri_label[start:end] if n != self.num_batchs-1 else fmri_label[start:])
            elif len(fmri_label) % self.batch_size ==0:
                self.batchset['fmri'].append(fmri_data[start:end])
                self.batchset['labels'].append(fmri_label[start:end])

        del fmri_data, fmri_label