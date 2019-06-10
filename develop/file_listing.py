import numpy as np
import os
import re
import random

def dir_search(basedir, in_condition_1=None, in_condition_2=None, in_condition_3=None, out_conidtion=None):
    
    search_list = []
    if in_condition_1:
            in_condition_1 = "|".join(in_condition_1).lower()
    if in_condition_2:
            in_condition_2 = "|".join(in_condition_2).lower()
    if in_condition_3:
            in_condition_3 = "|".join(in_condition_3).lower()
    if out_conidtion:
            out_conidtion = "|".join(out_conidtion).lower()
    
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

        search_list.append(dirpath)
        
    search_list.sort()
    search_list = np.array(search_list)
    
    return search_list

def beta_file_list(search_dir, choice_num):
    max_len = 0
    min_len = 0
    beta_full_path_list = []
    
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
            beta_full_path_list.append(os.path.join(dirpath,files))
    
    beta_full_path_list = np.array(beta_full_path_list)
    return beta_full_path_list

def beta_file_dict(beta_full_path_list):
    file_dict = {}
    temp_list = []
    for i in range(len(beta_full_path_list)):
        split_name = beta_full_path_list[i].split('/')

        if 'subject' in split_name[-2]:
            dic_key = split_name[-2]

        try:
            file_dict[dic_key]
        except:
            file_dict[dic_key] = [beta_full_path_list[i]]

        if file_dict[dic_key]:
            file_dict[dic_key].append(beta_full_path_list[i])
    return file_dict

def rating_file_list(basedir, in_condition_1=None, in_condition_2=None, in_condition_3=None, out_conidtion=None):
    search_list = dir_search(basedir=basedir, in_condition_1=in_condition_1, 
                     in_condition_2 = in_condition_2, in_condition_3=in_condition_3,
                     out_conidtion=out_conidtion)

    rating_full_path_list = []
    for dirpath in search_list:
        rating_files = os.listdir(dirpath)
        rating_files.sort()

        for file in rating_files:
            rating_full_path_list.append(os.path.join(dirpath,file))
    
    rating_full_path_list = np.array(rating_full_path_list)
    return rating_full_path_list

def rating_index_dict(beta_full_path_list, rating_full_path_list):

    index_dict = {}
    index_dict['beta_full_path_list'] = beta_full_path_list

    index_list = []
    for dirpath in beta_full_path_list:
        file_name = os.path.split(dirpath)[-1]
        c = ""
        for i in range(len(file_name)):
            if file_name[i].isnumeric():
                c = c+file_name[i]
        index_list.append(int(c))
    index_list_np = np.array(index_list) - 1

    index_dict['index_list_np'] = index_list_np

    return index_dict

def rating_value_list(rating_full_path_list, index_dict):
    if not len(index_dict['beta_full_path_list'])%len(rating_full_path_list) == 0:
        raise ValueError("beta_full_path_list length({}) have to be multiple of rating_full_path_list length({})".format(len(beta_full_path_list),len(rating_full_path_list)))
    
    if not len(index_dict['beta_full_path_list']) == len(index_dict['index_list_np']):
        raise ValueError("beta_full_path_list length({}) have to be matched with index_list_np length({})".format(len(beta_full_path_list), len(index_list_np)))
    
    index = 0
    index_2 = 0
    rating_list = []
    for dirpath in rating_full_path_list:
        rating_np = np.load(dirpath)  
        
        for i in range(int(len(index_dict['beta_full_path_list'])/len(rating_full_path_list))):
            check_subj = False
            for i in range(len(index_dict['beta_full_path_list'][index].split('/'))):
                if 'subject' in index_dict['beta_full_path_list'][index].split('/')[i]:
                    subj_name = index_dict['beta_full_path_list'][index].split('/')[i]
                    check_subj = True

            if not check_subj:
                raise ValueError("Cannot find subject name in index")

            if not subj_name in rating_full_path_list[index_2]:
                raise ValueError("index's subject name {} is not matched with rating_full_path_list's subject name {}".format(subj_name, rating_full_path_list[index_2]))

            rating_list.append(rating_np[index_dict['index_list_np'][index]])
            
            index += 1
        index_2 += 1
        
    rating_list = np.array(rating_list)
    return rating_list