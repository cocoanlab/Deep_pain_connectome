from scripts.SEMIC import *
from models import inception_v4
from sklearn import metrics

import tensorflow as tf

semic = load_dataset(workers=20, shuffle=True, batch_size = 20)

net = inception_v4.create(data_shape=(79, 95, 79, 1), num_output=2, mode='classification',optimizer_type='adadelta', phase='train')

num_epochs = 1000
lowest_loss=None

with net.sess as sess:
    with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        summary_writer = tf.summary.FileWriter("./out/", net.sess.graph)
        
        for epoch in range(num_epochs):

            prediction = {phase : [] for phase in ['train', 'valid']}
            loss = {phase : 0 for phase in ['train', 'valid']}
            count = {phase : 0 for phase in ['train', 'valid']}
            metric = {method : 0 for method in ['auc', 'sensitivity', 'specitivity']}
            for method in metric.keys():
                metric[method] = {phase : 0 for phase in ['train', 'valid']}
            
            for phase in ['train', 'valid']:
                for _ in range(len(semic.train_subj_list if phase == 'train' else semic.valid_subj_list)):
                    semic.load(phase)

                    for i in range(len(semic.batchset['labels'])):
                        pred, cost, op = sess.run([net.output, net.loss,net.train_op], 
                                                  feed_dict={net.x: semic.batchset['fmri'][i],
                                                             net.y: semic.batchset['labels'][i], 
                                                             net.lr : 1.,
                                                             net.keep_prob: 0.7,
                                                             net.is_train: True})

                        pred = pred.argmax(-1)
                        fpr, tpr, _ = metrics.roc_curve(semic.batchset['labels'][i], pred, pos_label=1)
                        tn, fp, fn, tp = metrics.confusion_matrix(semic.batchset['labels'][i], pred).ravel()
                        
                        metric['auc'][phase] += metrics.auc(fpr, tpr)
                        metric['sensitivity'][phase] += tp / (tp+fn)
                        metric['sensitivity'][phase] += tn / (tn+fp)
                        loss[phase] += cost
                        count[phase] += 1
            
            summ = tf.Summary()
            for phase in ['train', 'valid']:
                summ.value.add(tag=phase+'_loss', 
                               simple_value=loss[phase]/count[phase] if count[phase] !=0 else 0)
                
                for method in ['auc', 'sensitivity', 'specitivity']:
                    summ.value.add(tag=phase+'_'+method, 
                                   simple_value=metric[method][phase]/count[phase] if count[phase] !=0 else 0)
                    
            summary_writer.add_summary(summ,epoch)
            
            if epoch > 0:
                if lowest_loss == None or lowest_loss > loss['valid']/count['valid'] :
                    lowest_loss = loss['valid']/count['valid']
                    saver.save(sess, "./out/SEMIC/inception.ckpt")
