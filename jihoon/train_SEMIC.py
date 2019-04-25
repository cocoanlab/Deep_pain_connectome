from scripts.SEMIC import *
from models import inception_v4
from sklearn import metrics

import tensorflow as tf

semic = load_dataset(workers=20, shuffle=True, batch_size = 20)

net = inception_v4.create(data_shape=(79, 95, 79, 1), num_output=2, mode='classification',optimizer_type='adadelta', phase='train')

num_epochs = 1000
lowest_loss=None
highest_auc=None

output_list = [net.output, net.loss]

with net.sess as sess:
    with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        summary_writer = tf.summary.FileWriter("./out/", net.sess.graph)
        
        for epoch in range(num_epochs):

            feed_dict = {phase : {} for phase in ['train', 'valid']}
            prediction = {phase : [] for phase in ['train', 'valid']}
            loss = {phase : 0 for phase in ['train', 'valid']}
            count = {phase : 0 for phase in ['train', 'valid']}
            auc = {phase : 0 for phase in ['train', 'valid']}
            
            for phase in ['train', 'valid']:
                for _ in range(len(semic.train_subj_list if phase == 'train' else semic.valid_subj_list)):
                    semic.load(phase)
                    
                    for i in range(len(semic.batchset['labels'])):
                        feed_dict = {net.x: semic.batchset['fmri'][i],
                                     net.y: semic.batchset['labels'][i],
                                     net.keep_prob: 0.7 if phase=='train' else 1.0,
                                     net.is_train : True if phase=='train' else False}
                        
                        if phase == 'train' :
                            feed_dict.update({net.lr : 1. })
                            pred, cost, _ = sess.run(output_list+[net.train_op], feed_dict=feed_dict)
                        else :
                            pred, cost = sess.run(output_list, feed_dict=feed_dict)

                        pred = pred.argmax(-1)
                        fpr, tpr, _ = metrics.roc_curve(semic.batchset['labels'][i], pred, pos_label=1)
                        
                        auc[phase] += metrics.auc(fpr, tpr)
                        loss[phase] += cost
                        count[phase] += 1
            
            summ = tf.Summary()
            for phase in ['train', 'valid']:
                summ.value.add(tag=phase+'_loss', 
                               simple_value=loss[phase]/count[phase] if count[phase] !=0 else 0)
                summ.value.add(tag=phase+'_auc', 
                               simple_value=auc[phase]/count[phase] if count[phase] !=0 else 0)
                    
            summary_writer.add_summary(summ,epoch)
            
            if epoch > 0 and phase == 'valid':
                if lowest_loss == None or owest_loss > loss[phase]/count[phase] :
                    lowest_loss = loss[phase]/count[phase]
                    saver.save(sess, "./out/SEMIC/inception_lowest_loss.ckpt")
                if highest_auc == None or highest_auc < auc[phase]/count[phase] :
                    lowest_loss = auc[phase]/count[phase]
                    saver.save(sess, "./out/SEMIC/inception_highest_auc.ckpt")
