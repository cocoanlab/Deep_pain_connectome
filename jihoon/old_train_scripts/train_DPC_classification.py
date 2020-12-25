import sys ; sys.path.append('../')
import tensorflow as tf

from scripts.DPC import *
from models import SE_Dense_Resnet
from sklearn.metrics import accuracy_score

ckpt_path = "./out/DPC/"
phase_list = ['train', 'valid', 'test']

dpc = load_dataset(batch_size=100, split_mode='kfold', save_division_info=True, 
                   save_batchset=False, restore=False, num_total_K=5)
for K in range(5):
    if K > 0 : dpc.change_K_fold(test_K=str(K+1), reload=True)
    K = str(K+1)
    
    tf.reset_default_graph()
    
    net = SE_Dense_Resnet.create((79, 95, 68, 1), 6, conv_mode='3d', optimizer_type='adadelta',reduction_ratio=4, task='classification')

    num_epochs = 50
    lowest_loss=None
    highest_acc=None

    with net.sess as sess:
        with tf.device('/gpu:0'):
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            summary_writer = tf.summary.FileWriter(ckpt_path+'K{}/'.format(K), net.sess.graph)

            for epoch in range(num_epochs):

                feed_dict = {phase : {} for phase in phase_list}

                loss = {phase : 0 for phase in phase_list}
                count = {phase : 0 for phase in phase_list}
                acc = {phase : 0 for phase in phase_list}

                for phase in phase_list:
                    prediction = []
                    answer = []
                    for idx in range(len(dpc.batchset[phase]['label'])):
                        feed_dict = {net.x: dpc.batchset[phase]['fmri'][idx],
                                     net.y: dpc.batchset[phase]['label'][idx],
                                     net.keep_prob: 0.7 if phase=='train' else 1.0,
                                     net.is_train : True if phase=='train' else False}

                        if phase == 'train' :
                            feed_dict.update({net.lr : 1e-2 })
                            pred, cost, _ = sess.run([net.output, net.loss, net.train_op], feed_dict=feed_dict)
                        else :
                            pred, cost = sess.run([net.output, net.loss], feed_dict=feed_dict)

                        prediction.append(pred.argmax(-1))
                        answer.append(dpc.batchset[phase]['label'][idx])
                        loss[phase] += cost
                        count[phase] += 1

                    prediction = np.concatenate(prediction, axis=0)
                    answer = np.concatenate(answer, axis=0)
                    acc[phase] = accuracy_score(answer, prediction)

                summ = tf.Summary()
                for phase in phase_list:
                    summ.value.add(tag=phase+' loss', 
                                   simple_value=loss[phase]/count[phase] if count[phase] !=0 else 0)
                    summ.value.add(tag=phase+' Accuracy', 
                                   simple_value=acc[phase] if count[phase] !=0 else 0)

                summary_writer.add_summary(summ, epoch)

                if epoch > 0 and phase == 'test':
                    if lowest_loss == None or lowest_loss > loss[phase]/count[phase] :
                        lowest_loss = loss[phase]/count[phase]
                        saver.save(sess, ckpt_path+"K{}/SEDR_lowest_loss.ckpt".format(K))
                    if highest_acc == None or highest_acc < acc[phase] :
                        highest_acc = acc[phase]
                        saver.save(sess, ckpt_path+"K{}/SEDR_highest_acc.ckpt".format(K))
