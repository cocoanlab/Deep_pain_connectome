import warnings ; warnings.filterwarnings('ignore')
import sys; sys.path.append('../')
import os, gc, pickle, glob
import numpy as np
import nibabel as nib
import tensorflow as tf

from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask, unmask
from nilearn.image import smooth_img, resample_img
from multiprocessing import Pool
from tqdm import tqdm
from models import Residual_3D_CVAE as CVAE
from sklearn.metrics import pairwise_distances

class HCP:
    def __init__(self, HCP_dataset_path, batch_size, N_fold=10, seed=201703,
                 template_nii=None):
        if template_nii is not None:
            self.template = nib.load(template_nii)
        
        np.random.seed(seed)
        np.random.shuffle(HCP_dataset_path)
        
        self.batch_size = batch_size
        self.Kfold_path = np.array_split(HCP_dataset_path,N_fold)
        
        self.batchset = {}
        
        __dump_dat = pickle.load(open(self.Kfold_path[0][0],'rb'))
        self.main_tasks = sorted(set([task.split('_')[0] for task in __dump_dat.keys()]))
        self.sub_tasks = sorted(set([subtask for task in __dump_dat.values() for subtask in list(task.keys())]))
        self.total_sample_num = 0
        self.total_batch_num = 0
    
    def _count_num_data(self,path):
        with open(path,'rb') as fin:
            total_len = sum([1 for dat in pickle.load(fin).values() for _ in dat.values()])
            
        idx_start = list(range(0,total_len,self.batch_size))[:-1]
        idx_end = list(range(0,total_len,self.batch_size))[1:]
        cut_s_e = [[s,e] for s,e in zip(idx_start,idx_end)]
        if idx_end[-1] < total_len:
            cut_s_e.append([idx_end[-1],total_len])
        return total_len, len(cut_s_e)
            
    def count_num_batch(self, validset_idx=None, n_jobs=36):
        if  validset_idx is not None :
            train_path = [self.Kfold_path[i] for i in range(len(self.Kfold_path)) if i != validset_idx]
            train_path = np.concatenate(train_path)
        else :
            train_path = np.concatenate(self.Kfold_path)
        mp = Pool(processes=n_jobs)
        print('counting number of training batchset ...')
        
        total_sample_cnt = []
        total_batch_cnt = []
        pbar = tqdm(total=len(train_path))
        def update(result):
            total_sample_cnt.append(result[0])
            total_batch_cnt.append(result[1])
            pbar.update(1)
        
        
        for path in train_path:
            mp.apply_async(self._count_num_data, args=(path,), callback=update)

        mp.close()
        mp.join()
        self.total_sample_num = sum(total_sample_cnt)
        self.total_batch_num = sum(total_batch_cnt)
        print(f'total number of samples = {self.total_sample_num}')
        print(f'total number of batch = {self.total_batch_num}')
        
    def load(self, path):
        img_list = []
        lbl_list = []
        with open(path,'rb') as fin:
            for dat in pickle.load(fin).values():
                for task, fmri in dat.items():
                    fmri = resample_img(fmri, self.template.get_affine(), self.template.shape)
                    img_list.append(fmri.get_fdata()[np.newaxis])
                    lbl_list.append(self.sub_tasks.index(task))
                    if fmri.in_memory : 
                        fmri.uncache()
                        del fmri
                        gc.collect()
        return np.vstack(img_list), np.array(lbl_list)
        
    def load_data(self, k_idx, n_jobs=36):
        self.batchset.clear()
        train_path = self.Kfold_path[k_idx]
        fmri = []
        task = []
        
        mp = Pool(processes=n_jobs)
        pbar = tqdm(total=len(train_path))
        
        def update(result):
            fmri.append(result[0])
            task.append(result[1])
            pbar.update(1)

        for path in train_path:
            mp.apply_async(self.load, args=(path,), callback=update)

        mp.close()
        mp.join()
        pbar.close()
        
        self.batchset['fmri'] = np.vstack(fmri)
        self.batchset['task'] = np.concatenate(task)
        fmri.clear()
        task.clear()
        del fmri,task
        gc.collect()
        
#HCP_dataset_path = sorted(glob.glob('/media/das/HCP_run_beta/*'))
N_fold=10
HCP_dataset_path = sorted(glob.glob('/mnt/Volume1/HCP_run_beta/*'))
hcp_data = HCP(HCP_dataset_path, 8, N_fold, template_nii='./dataset_MPC/sub-mpc001/beta_0002.nii')
validset_idx=9
#hcp_data.count_num_batch(0,48)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='2'

tf.reset_default_graph()
#lr_config = CVAE.lr_scheduler_config(1e-2, 1e-7, 10, 2, 8, hcp_data.total_batch_num)
#net = CVAE.create(data_shape=(79,95,79,1), latent_size=128, lr_scheduler_config=lr_config)
net = CVAE.create(data_shape=(79,95,79,1), latent_size=128, num_cond=len(hcp_data.sub_tasks),
                  lr_init=1e-5, grad_threshold=1.)#

num_epochs = 100
lowest_loss = None
highest_corr = None

ckpt_path = f"./out/Residual_CVAE_K{validset_idx}/"
if os.path.isdir(ckpt_path):
    os.system(f'rm -rf {ckpt_path}')
    
Kfold_order = [i for i in range(N_fold) if i!=validset_idx]+[validset_idx]
hcp_data.batch_size=8
with tf.device('/gpu:0'):
    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(ckpt_path, net.sess.graph)

    for iteration in range(num_epochs):
        for k_idx in Kfold_order:
            summ = tf.Summary()
            Corr_total = []
            KL_total = []
            MSE_total = []
            
            phase = 'valid' if k_idx== validset_idx else 'train'
            hcp_data.load_data(k_idx, n_jobs=48)
            
            idx_start = list(range(0,len(hcp_data.batchset['fmri']),hcp_data.batch_size))[:-1]
            idx_end = list(range(0,len(hcp_data.batchset['fmri']),hcp_data.batch_size))[1:]
            cut_s_e = [[s,e] for s,e in zip(idx_start,idx_end)]
            if idx_end[-1] < len(hcp_data.batchset['fmri']):
                cut_s_e.append([idx_end[-1],len(hcp_data.batchset['fmri'])])
                
            with tf.device('/gpu:0'):
                for s, e in cut_s_e:
                    fmri_input = np.expand_dims(hcp_data.batchset['fmri'][s:e],-1)
                    feed_dict = {net.x : fmri_input,
                                 net.cond: hcp_data.batchset['task'][s:e],
                                 net.keep_prob: 0.7 if phase=='train' else 1.0,
                                 net.is_train : True if phase=='train' else False}
                    step, cost, _ = net.sess.run([net.global_step, net.loss, net.train_op], feed_dict=feed_dict)
                    
                    if phase == 'train' :
                        step, pred, kl, mse, _ = net.sess.run([net.global_step, net.output, net.KL_divergence, net.recon_loss, net.train_op], feed_dict=feed_dict)
                    else :
                        step, pred, kl, mse = net.sess.run([net.global_step, net.output, net.KL_divergence, net.recon_loss], feed_dict=feed_dict)
                        
                    fmri_input = np.squeeze(fmri_input).reshape(e-s,79*95*79)
                    pred = np.squeeze(pred).reshape(e-s,79*95*79)
                    corr = pairwise_distances(fmri_input,pred,'correlation', 48).diagonal().mean()
                    Corr_total.append(corr)
                    KL_total.append(kl)
                    MSE_total.append(mse)
                    
            Corr_total = np.mean(Corr_total)
            KL_total = np.mean(KL_total)
            MSE_total = np.mean(MSE_total)
            
            summ.value.add(tag=phase+' fold mean Correlation', simple_value=Corr_total)
            summ.value.add(tag=phase+' fold mean loss', simple_value=KL_total+MSE_total)
            summ.value.add(tag=phase+' fold KL', simple_value=KL_total)
            summ.value.add(tag=phase+' fold reconstruction error', simple_value=MSE_total)
            summary_writer.add_summary(summ, (k_idx*iteration)+(k_idx+1))
            
            if phase == 'valid':
                total_loss=KL_total+MSE_total
                if lowest_loss == None or lowest_loss > total_loss :
                    lowest_loss = total_loss
                    saver.save(net.sess, ckpt_path+"lowest_loss.ckpt")
                if highest_corr == None or highest_corr < Corr_total :
                    highest_corr = Corr_total
                    saver.save(net.sess, ckpt_path+"highest_correlation.ckpt")