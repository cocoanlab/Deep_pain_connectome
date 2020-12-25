import warnings ; warnings.filterwarnings('ignore')
import sys; sys.path.append('../')
import os, gc, pickle
import numpy as np
import nibabel as nib
import tensorflow as tf

from scipy.stats import pearsonr
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask, unmask
from multiprocessing import Pool
from tqdm import tqdm
from models import Convolutional_Autoencoder as CAE
from scripts.HCP import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

dataset_root = '/media/das/Human_Connectome_Project/'
ckpt_path = "./out/3D_CAE_Gaussian_Noise/"
_ = os.system(f'rm -rf {ckpt_path}')

hcp_data = HCP(dataset_root, batch_size = 4, load_size = 40, workers = 10, split_2d_at=None, shuffle=True,
               mask_path='../masks/dartel_spm12_mni152_brainmask.img', recon_mask=True, gaussian_noise=True)

tf.reset_default_graph()
phase_list = ['train', 'valid']

net = autoencoder = CAE.create(data_shape=(79,95,68,1), latent_size=256, optimizer_type='adadelta', enable_sc=True)

num_epochs = 1000
lowest_loss = None
highest_corr = None

with tf.device('/gpu:0'):
    net.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(ckpt_path, net.sess.graph)

    iteration = 0
    while True:

        feed_dict = {phase : {} for phase in phase_list}

        loss = {phase : 0 for phase in phase_list}
        corr = {phase : 0 for phase in phase_list}
        count = {phase : 0 for phase in phase_list}

        for phase in phase_list:
            prediction = []
            answer = []

            hcp_data.load_batchset(phase)
            for idx in range(len(hcp_data.batchset['fmri'])):
                batch_length = len(hcp_data.batchset['fmri'][idx])
                feed_dict = {net.x: hcp_data.batchset['fmri'][idx],
                             net.keep_prob: 0.7 if phase=='train' else 1.0,
                             net.is_train : True if phase=='train' else False}

                if phase == 'train' :
                    feed_dict.update({net.lr : 1. })
                    pred, cost, _ = net.sess.run([net.output, net.loss, net.train_op], feed_dict=feed_dict)
                else :
                    pred, cost = net.sess.run([net.output, net.loss], feed_dict=feed_dict)

                ans = hcp_data.batchset['fmri'][idx].reshape(-1)
                pred = pred.reshape(-1)
                
                corr[phase] += pearsonr(ans,pred)[0]
                loss[phase] += cost
                count[phase] += 1

            summ = tf.Summary()
            summ.value.add(tag=phase+' loss', simple_value=loss[phase]/count[phase] if count[phase] !=0 else 0)
            summ.value.add(tag=phase+' corr', simple_value=corr[phase]/count[phase] if count[phase] !=0 else 0)

            summary_writer.add_summary(summ, iteration)
            if phase == 'valid':
                if lowest_loss == None or lowest_loss > loss['valid']/count['valid'] :
                    lowest_loss = loss[phase]/count[phase]
                    saver.save(net.sess, ckpt_path+"lowest_loss.ckpt")
                    
                if highest_corr == None or highest_corr < corr['valid']/count['valid'] :
                    highest_corr = corr['valid']/count['valid']
                    saver.save(net.sess, ckpt_path+"highest_corr.ckpt")

        iteration+=1