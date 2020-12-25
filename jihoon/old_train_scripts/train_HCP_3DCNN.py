import warnings ; warnings.filterwarnings('ignore')
import sys; sys.path.append('../')
import os, gc, pickle
import numpy as np
import nibabel as nib
import tensorflow as tf

from sklearn.metrics import accuracy_score
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask, unmask
from multiprocessing import Pool
from tqdm import tqdm
from models import Vanila_3D_CNN as CNN
from scripts.HCP import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

dataset_root = '/media/das/Human_Connectome_Project/'
ckpt_path = "./out/Vanila_3D_CNN_v2/"

hcp_data = HCP(dataset_root, batch_size = 8, load_size = 40, workers = 20, split_2d_at=None, recon_mask=True)

train_mode = 'classification'

net = CNN.create(data_shape=(79,95,68,1), num_output=7, mode=train_mode, optimizer_type='adadelta')

tf.reset_default_graph()
phase_list = ['train', 'valid']

num_epochs = 1000
lowest_loss = None
highest_acc = None

with net.sess as sess:
    with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        summary_writer = tf.summary.FileWriter(ckpt_path, net.sess.graph)

        iteration = 0
        while True:
            
            feed_dict = {phase : {} for phase in phase_list}

            loss = {phase : 0 for phase in phase_list}
            count = {phase : 0 for phase in phase_list}
            summ = tf.Summary()
            
            for phase in phase_list:
                prediction = []
                answer = []

                hcp_data.load_batchset(phase)
                for idx in range(len(hcp_data.batchset['fmri'])):
                    batch_length = len(hcp_data.batchset['fmri'][idx])
                    feed_dict = {net.x: hcp_data.batchset['fmri'][idx],
                                 net.y: hcp_data.batchset['task'][idx],
                                 net.keep_prob: 0.7 if phase=='train' else 1.0,
                                 net.is_train : True if phase=='train' else False}

                    if phase == 'train' :
                        feed_dict.update({net.lr : 1. })
                        pred, cost, _ = sess.run([net.logits, net.loss, net.train_op], feed_dict=feed_dict)
                    else :
                        pred, cost = sess.run([net.logits, net.loss], feed_dict=feed_dict)
                    
                    if train_mode == 'classification':
                        prediction += list(pred.argmax(1))
                    
                    loss[phase] += cost
                    count[phase] += 1
                        
                total_mean_loss = loss[phase]/count[phase] if count[phase] !=0 else 0
                summ.value.add(tag=phase+' loss', simple_value=total_mean_loss)
                if train_mode == 'classification':
                    total_answer = list(np.concatenate(hcp_data.batchset['task']))
                    total_acc = accuracy_score(total_answer, prediction)
                    summ.value.add(tag=phase+' Accuracy', simple_value=total_acc)

            summary_writer.add_summary(summ, iteration)
            if phase == 'valid':
                if lowest_loss == None or lowest_loss > total_mean_loss :
                    lowest_loss = total_mean_loss
                    saver.save(sess, ckpt_path+"lowest_loss.ckpt")
                if train_mode == 'classification':
                    if highest_acc == None or highest_acc < total_acc :
                        highest_acc = total_acc
                        saver.save(sess, ckpt_path+"highest_accuracy.ckpt")

            iteration+=1