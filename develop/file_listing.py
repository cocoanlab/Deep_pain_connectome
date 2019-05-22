import numpy as np
import os
import re
import random

def dir_search(basedir, in_condition_1=None, in_condition_2=None, in_condition_3=None, out_conidtion=None):
    
    search_dir = []
    
    for dirpath, dirnames, filenames in os.walk(basedir):
        if in_condition_1:
            in_search_1 = re.search(in_condition_1, dirpath)
            if not in_search_1: 
                continue
        
        if in_condition_2:
            in_search_2 = re.search(in_condition_2, dirpath)
            if not in_search_2: 
                continue
                
        if in_condition_3:
            in_search_3 = re.search(in_condition_3, dirpath)
            if not in_search_3: 
                continue

        if out_conidtion:
            out_search = re.search(out_conidtion, dirpath)    
            if out_search:
                continue

        search_dir.append(dirpath)
        
    search_dir.sort()
    return search_dir

def file_list(search_dir, choice_num):
    max_len = 0
    min_len = 0
    full_path_list = []
    
    for dirpath in search_dir:
        if max_len < len(os.listdir(dirpath)):
            max_len = len(os.listdir(dirpath))

        if min_len == 0 or min_len > len(os.listdir(dirpath)):
            min_len = len(os.listdir(dirpath))

    if choice_num > min_len:
        raise ValueError("choice_num {} is larger than directory files minimum length {}".format(choice_num, min_len))

        
    for dirpath in search_dir:
        rand = np.random.choice(len(os.listdir(dirpath)), choice_num, replace=False) + 1
        file_name_list = []
        for i in range(rand.shape[0]):
            if rand[i]<10:
                name = 'beta000'+str(rand[i])
            elif rand[i]<100:
                name ='beta00'+str(rand[i])
            elif rand[i]<1000:
                name ='beta0'+str(rand[i])
            else:
                name = 'beta'+str(rand[i])

            file_name = name + '.npy'
            file_name_list.append(file_name)

        for files in file_name_list:
            full_path_list.append(os.path.join(dirpath,files))
            
    return full_path_list

def file_dict(full_path_list):
    file_dict = {}
    temp_list = []
    for i in range(len(full_path_list)):
        split_name = full_path_list[i].split('/')

        if 'subject' in split_name[-2]:
            dic_key = split_name[-2]

        try:
            file_dict[dic_key]
        except:
            file_dict[dic_key] = [full_path_list[i]]

        if file_dict[dic_key]:
            file_dict[dic_key].append(full_path_list[i])
    return file_dict
