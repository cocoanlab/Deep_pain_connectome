import sys ; sys.path.append('../')
import tensorflow as tf

from scripts.DPC import *
from models import SE_Dense_Resnet
from sklearn.metrics import r2_score

ckpt_path = "./out/DPC/"
phase_list = ['train', 'valid', 'test']

dpc = load_dataset(shuffle=True, batch_size=100)
net = SE_Dense_Resnet.create((79, 95, 68, 1), 1, conv_mode='3d', optimizer_type='adam',reduction_ratio=4, mode='regression')

num_epochs = 10000
lowest_loss=None
highest_rsq=None

with net.sess as sess:
    with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        summary_writer = tf.summary.FileWriter(ckpt_path, net.sess.graph)
        
        for epoch in range(num_epochs):

            feed_dict = {phase : {} for phase in phase_list}
            
            loss = {phase : 0 for phase in phase_list}
            count = {phase : 0 for phase in phase_list}
            rsq = {phase : 0 for phase in phase_list}
            
            for phase in phase_list:
                prediction = []
                answer = []
                for idx in range(len(dpc.batchset[phase]['rating'])):
                    feed_dict = {net.x: dpc.batchset[phase]['fmri'][idx],
                                 net.y: dpc.batchset[phase]['rating'][idx],
                                 net.keep_prob: 0.7 if phase=='train' else 1.0,
                                 net.is_train : True if phase=='train' else False}

                    if phase == 'train' :
                        feed_dict.update({net.lr : 1e-2 })
                        pred, cost, _ = sess.run([net.output, net.loss, net.train_op], feed_dict=feed_dict)
                    else :
                        pred, cost = sess.run([net.output, net.loss], feed_dict=feed_dict)

                    prediction.append(pred)
                    answer.append(dpc.batchset[phase]['rating'][idx])
                    loss[phase] += cost
                    count[phase] += 1

                prediction = np.concatenate(prediction, axis=0)
                answer = np.concatenate(answer, axis=0)
                rsq[phase] = r2_score(answer, prediction)

            summ = tf.Summary()
            for phase in phase_list:
                summ.value.add(tag=phase+'_loss', 
                               simple_value=loss[phase]/count[phase] if count[phase] !=0 else 0)
                summ.value.add(tag=phase+'_R-Squared', 
                               simple_value=rsq[phase] if count[phase] !=0 else 0)

            summary_writer.add_summary(summ, epoch)
            
            if epoch > 0 and phase == 'test':
                if lowest_loss == None or lowest_loss > loss[phase]/count[phase] :
                    lowest_loss = loss[phase]/count[phase]
                    saver.save(sess, ckpt_path+"SEIR_lowest_loss.ckpt")
                if highest_rsq == None or highest_rsq < rsq[phase] :
                    highest_rsq = rsq[phase]
                    saver.save(sess, ckpt_path+"SEIR_highest_rsq.ckpt")
