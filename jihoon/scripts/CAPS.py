import numpy as np
import SimpleITK as sitk
import os, glob
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

def load_dataset(dataset_path, batch_size= None, shuffle=False):
    fmri_data = []
    fmri_label = []

    with tqdm(total=len(dataset_path)) as pbar:
        pbar.set_description('[ Load CAPS FMRI Image data ]')

        for path in dataset_path:
            img_data = sitk.ReadImage(path)
            img_data = sitk.GetArrayFromImage(img_data)

            img_label = 1 if path.split('/')[3]=='caps' else 0
            img_label = np.array([img_label for _ in range(len(img_data))])

            fmri_data.append(img_data)
            fmri_label.append(img_label)
            pbar.update(1)
            
        fmri_data = np.concatenate(fmri_data, axis=0)
        fmri_label = np.concatenate(fmri_label, axis=0)
        
    if shuffle :
        indices = np.random.permutation(len(fmri_label))
        fmri_data = fmri_data[indices]
        fmri_label = fmri_label[indices]


    if batch_size != None:
        num_batchs = int(len(fmri_label)/batch_size)+1 if len(fmri_label) % batch_size !=0 else int(len(fmri_label)/batch_size)

        img_list = []
        label_list = []

        for n in range(num_batchs):
            if len(fmri_label) % batch_size !=0:
                img_list.append(fmri_data[n*batch_size:(n+1)*batch_size] if n != num_batchs-1 else fmri_data[n*batch_size:])
                label_list.append(fmri_label[n*batch_size:(n+1)*batch_size] if n != num_batchs-1 else fmri_label[n*batch_size:])
            elif len(fmri_label) % batch_size ==0:
                img_list(fmri_data[n*batch_size:(n+1)*batch_size])
                label_list.append(fmri_label[n*batch_size:(n+1)*batch_size])

        del fmri_data, fmri_label
        return img_list, label_list
    else : 
        return fmri_data, fmri_label