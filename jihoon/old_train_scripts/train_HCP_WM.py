import warnings ; warnings.filterwarnings('ignore')
import sys; sys.path.append('../')
import nibabel as nib
import pickle, os, gc
import tensorflow as tf
import numpy as np

from nilearn import plotting
from multiprocessing import Pool
from tqdm import tqdm
from models import Vanila_3D_CNN as CNN
from sklearn.metrics import accuracy_score
from nilearn.masking import apply_mask

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'

subtask_list = {
    'WM' : ['0bk_body','0bk_faces','0bk_tools','0bk_places',
            '2bk_body','2bk_faces','2bk_tools','2bk_places',]
}

total_classes = [v for k, v in subtask_list.items() ]
total_classes = [y for x in total_classes for y in x] 

dataset_path = './dataset_WM'
#dataset_path = '/media/cnir09/GPU1_sync/data/Deep_pain_connectome/HCP_WM_beta/'

np.random.seed(201703)
_, _, beta_path = next(os.walk(dataset_path))
beta_path = [os.path.join(dataset_path,bpath) for bpath in beta_path]
beta_path = np.random.choice(beta_path, 33, replace=False)

test_beta_path = np.random.choice(beta_path, 3, replace=False)
train_beta_path = [path for path in beta_path if path not in test_beta_path]

beta_path = {'train' : train_beta_path, 'test':test_beta_path}

phase_list = ['train', 'test']

fmri_img = {p : [] for p in phase_list}
fmri_task = {p : [] for p in phase_list}

def read_data(path, phase):
    fmri = []
    labels = []

    with open(path,'rb') as fin :
        beta = pickle.load(fin)
        for task_run in beta.keys():
            if 'WM' in task_run:
                for subtask in beta[task_run].keys():
                    fmri.append(beta[task_run][subtask][np.newaxis,:,:,:,np.newaxis])
                    labels.append(total_classes.index(subtask))

    beta.clear(); del beta; gc.collect()
    return phase, fmri, labels

def get_hcp_data(phase, batch_size=64, iteration=0):
    fmri_img[phase].clear()
    fmri_task[phase].clear()
    
    batch_img = []
    batch_lbl = []

    def update(result):
        phase, fmri, labels = result

        for img, lbl in zip(fmri, labels):
            batch_img.append(img)
            batch_lbl.append(lbl)

            if len(batch_img) == batch_size:
                fmri_img[phase].append(np.concatenate(batch_img,0))
                fmri_task[phase].append(np.array(batch_lbl))
                batch_img.clear()
                batch_lbl.clear()
        if len(batch_img) != 0:
            fmri_img[phase].append(np.concatenate(batch_img,0))
            fmri_task[phase].append(np.array(batch_lbl))
            batch_img.clear()
            batch_lbl.clear()

        del result; gc.collect()
        pbar.update(1)

    '''
    if phase == 'train':
        data_path_list = np.array_split(beta_path['train'],2)[iteration%2]
    else :
        data_path_list = beta_path['test']
    '''
    data_path_list = beta_path[phase]
    
    mp = Pool(60)
    pbar = tqdm(total=len(data_path_list))
    pbar.set_description(f'[ ITER:{iteration} | {phase} dataset ] loading... ')

    for path in data_path_list:
        mp.apply_async(read_data, args=(path,phase), callback=update)
    mp.close()
    mp.join()
    pbar.close()
    
phase_list = ['train', 'test']

num_epochs = 10000
lowest_loss = None
highest_acc = None

#ckpt_path = "./out/Vanila_CNN_vBETA_new1/"
ckpt_path = "./out/Vanila_CNN_vBETA_transfered/"
if os.path.isdir(ckpt_path):
    os.system(f'rm -rf {ckpt_path}')

tf.reset_default_graph()
#net = CNN.create(data_shape=(79,95,68,1), num_output=len(total_classes), mode='classification', optimizer_type='adam')

net = CNN.create(data_shape=(79,95,68,1), num_output=len(total_classes), ckpt_path = './out/HCP_pretrained/highest_accuracy.ckpt',
                 freeze_conv_layer=True, mode='classification', optimizer_type='adam')

get_hcp_data('test', batch_size=65)
get_hcp_data('train', batch_size=65)
    
train_mode = 'classification'
with tf.device('/gpu:0'):
    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(ckpt_path, net.sess.graph)

    for iteration in range(num_epochs):

        feed_dict = {phase : {} for phase in phase_list}

        loss = {phase : 0 for phase in phase_list}
        count = {phase : 0 for phase in phase_list}
        summ = tf.Summary()

        for phase in phase_list:
            prediction = []
            answer = []
            
            for imgs, labels in zip(fmri_img[phase],fmri_task[phase]):
                feed_dict = {net.x: imgs,
                             net.y: labels,
                             net.keep_prob: 0.7 if phase=='train' else 1.0,
                             net.is_train : True if phase=='train' else False}

                if phase == 'train' :
                    feed_dict.update({net.lr : 1e-3 })
                    pred, cost, _ = net.sess.run([net.logits, net.loss, net.train_op], feed_dict=feed_dict)
                else :
                    pred, cost = net.sess.run([net.logits, net.loss], feed_dict=feed_dict)

                if train_mode == 'classification':
                    prediction += list(pred.argmax(1))

                loss[phase] += cost
                count[phase] += 1
        
            total_mean_loss = loss[phase]/count[phase] if count[phase] !=0 else 0
            summ.value.add(tag=phase+' loss', simple_value=total_mean_loss)
            if train_mode == 'classification':
                total_answer = list(np.concatenate(fmri_task[phase],0))
                total_acc = accuracy_score(total_answer, prediction)
                summ.value.add(tag=phase+' Accuracy', simple_value=total_acc)

        summary_writer.add_summary(summ, iteration)
        if phase == 'test':
            if lowest_loss == None or lowest_loss > total_mean_loss :
                lowest_loss = total_mean_loss
                saver.save(net.sess, ckpt_path+"lowest_loss.ckpt")
            if highest_acc == None or highest_acc < total_acc :
                highest_acc = total_acc
                saver.save(net.sess, ckpt_path+"highest_accuracy.ckpt")