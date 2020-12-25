import sys ; sys.path.append('../')
import tensorflow as tf

from scripts.DPC import *
from models import SE_Dense_Resnet
from sklearn.metrics import accuracy_score, r2_score

ckpt_path = "./out/DPC/"
phase_list = ['train', 'valid', 'test']

dpc = load_dataset(batch_size=100, split_mode='kfold', save_division_info=True, 
                   save_batchset=False, restore=False, num_total_K=5, task = 'both')

for K in range(5):
    if K > 0 : dpc.change_K_fold(test_K=str(K+1), reload=True)
    K = str(K+1)
    
    tf.reset_default_graph()
    
    net = SE_Dense_Resnet.create((79, 95, 68, 1), 6, conv_mode='3d', 
                                 optimizer_type='adadelta',reduction_ratio=4, task='both')

    num_epochs = 51
    lowest_total_loss=None
    lowest_cls_loss=None
    lowest_reg_loss=None
    highest_acc=None
    highest_rsq=None

    with net.sess as sess:
        with tf.device('/gpu:0'):
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            summary_writer = tf.summary.FileWriter(ckpt_path+'K{}/'.format(K), net.sess.graph)

            for epoch in range(num_epochs):

                feed_dict = {phase : {} for phase in phase_list}

                loss_cls = {phase : 0 for phase in phase_list}
                loss_reg = {phase : 0 for phase in phase_list}
                loss_total = {phase : 0 for phase in phase_list}
                count = {phase : 0 for phase in phase_list}
                acc = {phase : 0 for phase in phase_list}
                rsq = {phase : 0 for phase in phase_list}

                for phase in phase_list:
                    prediction_cls = []
                    prediction_reg = []
                    answer_cls = []
                    answer_reg = []
                    
                    for idx in range(len(dpc.batchset[phase]['label'])):
                        feed_dict = {net.x: dpc.batchset[phase]['fmri'][idx],
                                     net.y_cls: dpc.batchset[phase]['label'][idx],
                                     net.y_reg: dpc.batchset[phase]['label'][idx],
                                     net.keep_prob: 0.7 if phase=='train' else 1.0,
                                     net.is_train : True if phase=='train' else False}

                        if phase == 'train' :
                            feed_dict.update({net.lr : 1e-2 })
                            pred_cls, pred_reg, cost_cls, cost_reg, cost_total, _ = sess.run(
                                [net.output_cls, net.output_reg, net.loss_cls, net.loss_reg, net.total_loss, net.train_op],
                                feed_dict=feed_dict)
                        else :
                            pred_cls, pred_reg, cost_cls, cost_reg, cost_total = sess.run(
                                [net.output_cls, net.output_reg, net.loss_cls, net.loss_reg, net.total_loss],
                                feed_dict=feed_dict)

                        prediction_cls.append(pred_cls.argmax(-1))
                        answer_cls.append(dpc.batchset[phase]['label'][idx])
                        
                        prediction_reg.append(pred_reg)
                        answer_reg.append(dpc.batchset[phase]['rating'][idx])
                        
                        loss_cls[phase] += cost_cls
                        loss_reg[phase] += cost_reg
                        loss_total[phase] += cost_total
                        
                        count[phase] += 1

                    prediction_cls = np.concatenate(prediction_cls, axis=0)
                    prediction_reg = np.concatenate(prediction_reg, axis=0)
                    
                    answer_cls = np.concatenate(answer_cls, axis=0)
                    answer_reg = np.concatenate(answer_reg, axis=0)
                    
                    acc[phase] = accuracy_score(answer_cls, prediction_cls)
                    rsq[phase] = r2_score(answer_reg, prediction_reg)
                    

                summ = tf.Summary()
                for phase in phase_list:
                    
                    total_loss = loss_total[phase]/count[phase]
                    cls_loss = loss_cls[phase]/count[phase]
                    reg_loss = loss_reg[phase]/count[phase]
                    
                    summ.value.add(tag=phase+' classification loss', 
                                   simple_value=cls_loss if count[phase] !=0 else 0)
                    summ.value.add(tag=phase+' regression loss', 
                                   simple_value=reg_loss if count[phase] !=0 else 0)
                    summ.value.add(tag=phase+' total loss', 
                                   simple_value=total_loss if count[phase] !=0 else 0)
                    summ.value.add(tag=phase+' Accuracy', 
                                   simple_value=acc[phase] if count[phase] !=0 else 0)
                    summ.value.add(tag=phase+' R-Squared', 
                                   simple_value=rsq[phase] if count[phase] !=0 else 0)

                summary_writer.add_summary(summ, epoch)

                if epoch > 0 and phase == 'test':
                    if lowest_total_loss == None or lowest_total_loss > total_loss :
                        lowest_total_loss = total_loss
                        saver.save(sess, ckpt_path+"K{}/SEDR_lowest_total_loss.ckpt".format(K))
                        
                    if lowest_cls_loss == None or lowest_cls_loss > cls_loss :
                        lowest_cls_loss = cls_loss
                        saver.save(sess, ckpt_path+"K{}/SEDR_lowest_cls_loss.ckpt".format(K))
                        
                    if lowest_reg_loss == None or lowest_reg_loss > reg_loss :
                        lowest_reg_loss = reg_loss
                        saver.save(sess, ckpt_path+"K{}/SEDR_lowest_reg_loss.ckpt".format(K))
                        
                    if highest_acc == None or highest_acc < acc[phase] :
                        highest_acc = acc[phase]
                        saver.save(sess, ckpt_path+"K{}/SEDR_highest_acc.ckpt".format(K))
                        
                    if highest_rsq == None or highest_rsq < rsq[phase] :
                        highest_rsq = rsq[phase]
                        saver.save(sess, ckpt_path+"K{}/SEDR_highest_rsq.ckpt".format(K))
