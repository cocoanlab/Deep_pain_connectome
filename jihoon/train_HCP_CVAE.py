import glob, os
import numpy as np
import sys; sys.path.append('../')
import tensorflow as tf
import threading

from tqdm import tqdm
from nilearn.image import load_img
from nilearn.image import resample_img, math_img
from nilearn.masking import apply_mask, unmask
from sklearn.metrics import pairwise_distances
#from models import CVAE_Res as VAE
#from models import CVAE_MI as VAE
#from models import CVAE_small as VAE
from models import CVAE as VAE
from scripts.utils import *
from scripts.HCP import HCP
import scipy.stats

batch_size = 10
N_fold=10
validset_idx=1
target_task = 'sub'

GM_PATH = '../masks/gray_matter_mask.nii'
target_affine = np.array([[  -2.,    0.,    0.,   78.],
                          [   0.,    2.,    0., -112.],
                          [   0.,    0.,    2.,  -70.],
                          [   0.,    0.,    0.,    1.]])
target_shape = (79,95,79)

gm_mask = load_img(GM_PATH)
gm_mask = resample_img(gm_mask, target_affine, target_shape)
gm_mask = math_img('img>0',img=gm_mask)

dataset_path = sorted(glob.glob('/media/das/Deep_Pain_Connectome/*/*/*'))
#dataset_path = sorted(glob.glob('/mnt/Volume1/Deep_Pain_Connectome/*/*/*'))
#dataset_path = sorted(glob.glob('/cocoanlab/Deep_Pain_Connectome/Deep_Pain_Connectome/*/*/*'))
hcp_data = HCP(dataset_path, batch_size, N_fold, target_task, template_nii='./dataset_MPC/sub-mpc001/beta_0002.nii',
              exclude_tasks=['RESTING',], verbose=False)

Kfold_order = [i for i in range(N_fold) if i!=validset_idx]+[validset_idx]

WHERE_PC='GPU5'.upper()
#WHERE_PC='CNIRGPU'.upper()
GPU_NUM = 0

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_NUM)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

n_jobs=20
SEED=201703
NUM_EPOCH = 10000

phase_list = ['train', 'valid']

latent_size=1024
lr_init=1e-3
act_type='relu'

#additional = 'PAIN'
#additional = 'gradNorm1_radam_scheduled'

#TRAIN_TAG = f'{WHERE_PC}-{GPU_NUM}_HCP_{target_task}_{latent_size}_B{KL_beta}_lr{lr_init}_{additional}'
TRAIN_TAG = f'{WHERE_PC}-{GPU_NUM}_HCP_{target_task}_{latent_size}_Original_WMSE/'
LOG_FILE_PATH = f'logs/CVAE_{TRAIN_TAG}/'
SAVE_FILE_PATH = f'out/CVAE_{TRAIN_TAG}/'

VAE_model = VAE.VAE((79,95,79,1),
                    latent_size=latent_size, 
                    phase='train', 
                    num_cond=len(hcp_data.sub_tasks) if target_task == 'sub' else len(hcp_data.main_tasks),
                    do_bn=False,
                    lr_init=lr_init,
                    act_type=act_type,
                    optimizer_type='adam',
                    grad_threshold=1,
                    mixed_precision=False,)

def write_metrics(writer, phase, step, metrics):
    loss_met, kl_met, pixelwise_loss_met, corr_met, lr_met, pc_met = metrics
    
    with writer.as_default():
        tf.summary.scalar(f'{phase}/total_loss', loss_met.result(), step=step)
        tf.summary.scalar(f'{phase}/KL', kl_met.result(), step=step)
        tf.summary.scalar(f'{phase}/pixelwise loss', pixelwise_loss_met.result(), step=step)
        tf.summary.scalar(f'{phase}/correlation', corr_met.result(), step=step)
        tf.summary.scalar(f'{phase}/posterior_collapse', pc_met.result(), step=step)
        '''
        if phase == 'train' :
            tf.summary.scalar(f'{phase}/learning_rate', lr_met.result(), step=step)
        '''    
    for m in metrics : 
        m.reset_states()

total_train_step = 0
total_valid_step = 0

train_iter_time = 0
valid_iter_time = 0

lowest_loss = None
highest_corr = None

_ = os.system(f'rm -rf {LOG_FILE_PATH}')
_ = os.system(f'rm -rf {SAVE_FILE_PATH}')
writer = tf.summary.create_file_writer(LOG_FILE_PATH)

pbar = tqdm(total=NUM_EPOCH)
pbar.set_description('[  VAE  ] Progress ')

def plot_recon_result(true_dataset, pred_dataset, writer, step, phase='Inference', mask=None):
    plot_stat_map_TB(true_dataset, pred_dataset, writer, step, mask=mask, display_mode='x', phase=phase)
    plot_stat_map_TB(true_dataset, pred_dataset, writer, step, mask=mask, display_mode='z', phase=phase)

fold_epoch = -1
do_lagging = False

for epoch in range(NUM_EPOCH):
    for k_idx in Kfold_order:
        fold_epoch+=1
        
        model_mu_result = []
        approx_mu_result = []
        
        loss_met, kl_met, pixelwise_loss_met, corr_met, lr_met, pc_met = metric_list = [tf.keras.metrics.Mean() for _ in range(6)]
        
        phase = 'valid' if k_idx== validset_idx else 'train'
        hcp_data.load_data(k_idx, n_jobs=n_jobs)
    
        with tf.device('/gpu:0'):
            for i in range(len(hcp_data.batchset['task'])):
                fmri = hcp_data.batchset['fmri'][i]
                task = hcp_data.batchset['task'][i]
                beta = 1
                do_lagging=False
                if phase == 'train':
                    mu, _, pred, loss, kl, pix_loss, corr, curr_lr = VAE_model.train(fmri, task, KL_beta=beta, lagging=do_lagging)
                    total_train_step+=1
                    if total_train_step % 100 == 0 : 
                        true_nii = numpy2nii(fmri)
                        pred_nii = tensor2nii(pred)
                        threading.Thread(target=plot_recon_result, 
                                         args=(true_nii, pred_nii, writer, total_train_step, phase, gm_mask,)).start()
                else :
                    mu, _, pred, loss, kl, pix_loss, corr = VAE_model(fmri,task)
                    total_valid_step+=1
                    if total_valid_step % 100 == 0 : 
                        true_nii = numpy2nii(fmri)
                        pred_nii = tensor2nii(pred)
                        threading.Thread(target=plot_recon_result, 
                                         args=(true_nii, pred_nii, writer, total_valid_step, phase, gm_mask,)).start()
                
                approx_dat = VAE_model.encoder_input(pred, task)
                approx_mu, _ = VAE_model.encoder(approx_dat)
                
                model_mu_result.append(mu.numpy())
                approx_mu_result.append(approx_mu.numpy())
                
                loss_met(loss)
                kl_met(kl)
                pixelwise_loss_met(pix_loss)
                corr_met(corr)
                lr_met(curr_lr)
                
            curr_mean_loss = loss_met.result()
            curr_mean_corr = corr_met.result()
            
            model_mu = np.concatenate(model_mu_result,0).reshape(-1)
            aprox_mu = np.concatenate(approx_mu_result,0).reshape(-1)
            
            posterior_collapes = 1 - scipy.stats.pearsonr(model_mu.reshape(-1),aprox_mu.reshape(-1))[0]
            pc_met(posterior_collapes)

            if phase == 'train':
                write_metrics(writer, phase, train_iter_time, metric_list)
                train_iter_time+=1
                
            else :
                write_metrics(writer, phase, valid_iter_time, metric_list)
                valid_iter_time+=1
                
                if lowest_loss == None or lowest_loss > curr_mean_loss :
                    lowest_loss = curr_mean_loss
                    threading.Thread(target=VAE_model.save_weights, 
                                     args=(os.path.join(SAVE_FILE_PATH,'lowest_loss.ckpt'),)).start()
                if highest_corr == None or highest_corr < curr_mean_corr :
                    highest_corr = curr_mean_corr
                    threading.Thread(target=VAE_model.save_weights, 
                                         args=(os.path.join(SAVE_FILE_PATH,'highest_correlation.ckpt'),)).start()

        pbar.update(1)
