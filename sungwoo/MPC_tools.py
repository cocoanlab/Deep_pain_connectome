import os
import numpy as np
import pandas as pd
import nibabel as nib
import timeit

def making_file_path(base_dir):

    dir_path_list = []
    filenames_list = []

    for dirpath, dirnames, filenames in os.walk(base_dir):
        if 'sub-semic' in dirpath:
            if not 'no_GM' in dirpath:
                dir_path_list.append(dirpath)

                temp_filenames_list=[]
                for i in range(len(filenames)):
                    if 'beta' and 'nii' in filenames[i]:
                        temp_filenames_list.append(filenames[i])
                filenames_list.append(temp_filenames_list)

    dir_path_list = np.array(dir_path_list)
    filenames_list = np.array(filenames_list)
    
    return dir_path_list, filenames_list

def making_full_path_list(dir_path_list, filenames_list):
    full_path_list = []

    for i in range(dir_path_list.shape[0]):
        temp_path_list=[]
        for j in range(len(filenames_list[i])):
            temp_path = os.path.join(dir_path_list[i], filenames_list[i][j])
            temp_path_list.append(temp_path)
        full_path_list.append(temp_path_list)

    full_path_list = np.array(full_path_list)
    
    return full_path_list

def load_nii(load_index, full_path_list, data_num):
    X = []
    Y = []

    start = timeit.default_timer()
    for index in load_index:
        for i in range(data_num):
            temp_nii = nib.load(full_path_list[index][i]).get_data()
            temp_nii = np.array(temp_nii[np.newaxis,:,:,:])

            if len(X) == 0:
                X = temp_nii
            else:
                X = np.concatenate((X, temp_nii), axis=0)

            if 'model02_FIR_SPM_SINGLE_TRIAL' in full_path_list[index][i]:
                label = 0
                Y.append(label)
            elif 'model02_Overall_FIR_SPM_SINGLE_TRIAL' in full_path_list[index][i]:
                label = 1
                Y.append(label)

            if i%100 ==0:
                current_dir = os.path.dirname(full_path_list[index][0])
                print("***  Finished Loading {}th Nii File In {}  ***".format(i+1, current_dir))

        stop = timeit.default_timer()
        check_time = stop-start
        print('\n')
        print('##########################################')
        print("###  Excecuted Time : {}  ###".format(check_time))
        print('##########################################')
        print('\n')

    X = np.nan_to_num(X)
    Y = np.array(Y)

    print("X shape is {} ".format(X.shape))
    print("Y shape is {} ".format(Y.shape))
    
    return X, Y

def flatten_nii(X):
    flatten_X = []

    start = timeit.default_timer()
    for i in range(X.shape[0]):
        temp_flatten_x = X[i,:,:,:].flatten()
        temp_flatten_x = temp_flatten_x[np.newaxis,:]
        if i == 0:
            flatten_X = temp_flatten_x
        else:
            flatten_X = np.concatenate((flatten_X, temp_flatten_x),axis=0)

        if i%100 == 0:
            print("***  Finished Flatten {}th Data  ***".format(i+1))
            stop = timeit.default_timer()
            check_time = stop-start
            print("###  Excecuted Time : {}  ###".format(check_time))
            print('\n')

    flatten_X = pd.DataFrame(flatten_X)
    print("Result's shape is {} ".format(flatten_X.shape))
    
    return flatten_X