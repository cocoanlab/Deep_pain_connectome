import sys ; sys.path.append('../')
import tensorflow as tf

from scripts.SEMIC import *
from models import SE_Inception_ResNet_v4
#from models import Dense_Resnet
#from models import SE_Dense_Resnet
from sklearn.metrics import roc_curve, confusion_matrix, auc

ckpt_path = "./out/SEMIC/"
phase_list = ['train', 'valid', 'test']

semic = load_dataset(shuffle=True, batch_size = 30, nan_to_zero=True)

net = SE_Inception_ResNet_v4.create(data_shape=(79, 95, 79, 1), num_output=2, mode='classification',optimizer_type='adadelta', phase='train',reduction_ratio=4)
#net = SE_Dense_Resnet.create((79, 95, 79, 1), 2, conv_mode='3d', optimizer_type='adam',reduction_ratio=4)

num_epochs = 10000
lowest_loss=None
highest_auc=None

with net.sess as sess:
    with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        summary_writer = tf.summary.FileWriter(ckpt_path, net.sess.graph)
        
        for epoch in range(num_epochs):

            feed_dict = {phase : {} for phase in phase_list}
            
            loss = {phase : 0 for phase in phase_list}
            count = {phase : 0 for phase in phase_list}
            metrics = {phase : {} for phase in ['auc', 'sensitivity', 'specitivity']}
            for method in metrics.keys():
                metrics[method] = {phase : 0 for phase in phase_list}
            
            for phase in phase_list:
                prediction = []
                answer = []
                for _ in range(len(semic.batch_pathset[phase])):
                    semic.load(phase)
                    
                    feed_dict = {net.x: semic.fmri[:,:,:,:,np.newaxis],
                                 net.y: semic.labels,
                                 net.keep_prob: 0.7 if phase=='train' else 1.0,
                                 net.is_train : True if phase=='train' else False}

                    if phase == 'train' :
                        feed_dict.update({net.lr : 1. })
                        pred, cost, _ = sess.run([net.output, net.loss, net.train_op], feed_dict=feed_dict)
                    else :
                        pred, cost = sess.run([net.output, net.loss], feed_dict=feed_dict)

                    prediction.append(pred.argmax(-1))
                    answer.append(semic.labels)
                    loss[phase] += cost
                    count[phase] += 1

                prediction = np.concatenate(prediction, axis=0)
                answer = np.concatenate(answer, axis=0)

                fpr, tpr, _ = roc_curve(answer, prediction, pos_label=1)
                tn, fp, fn, tp = confusion_matrix(answer, prediction).ravel()
                metrics['auc'][phase] = auc(fpr, tpr)
                metrics['sensitivity'][phase] = tp / (tp+fn)
                metrics['specitivity'][phase] = tn / (tn+fp)

            summ = tf.Summary()
            for phase in phase_list:
                summ.value.add(tag=phase+'_loss', 
                               simple_value=loss[phase]/count[phase] if count[phase] !=0 else 0)
                for method in metrics.keys():
                    summ.value.add(tag=phase+'_'+method, 
                                   simple_value=metrics[method][phase] if count[phase] !=0 else 0)

            summary_writer.add_summary(summ, epoch)
            
            if epoch > 0 and phase == 'test':
                if lowest_loss == None or lowest_loss > loss[phase]/count[phase] :
                    lowest_loss = loss[phase]/count[phase]
                    saver.save(sess, ckpt_path+"SEDR_lowest_loss.ckpt")
                if highest_auc == None or highest_auc < metrics['auc'][phase] :
                    highest_auc = metrics['auc'][phase]
                    saver.save(sess, ckpt_path+"SEDR_highest_auc.ckpt")
