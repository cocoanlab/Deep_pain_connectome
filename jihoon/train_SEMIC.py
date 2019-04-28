from scripts.SEMIC import *
from models import inception_v4
from sklearn import metrics

import tensorflow as tf

ckpt_path = "./out/SEMIC/"
phase_list = ['train', 'valid']

semic = load_dataset(workers=20, shuffle=True, batch_size = 10)

net = inception_v4.create(data_shape=(79, 95, 79, 1), num_output=2, mode='classification',optimizer_type='adadelta', phase='train')

num_epochs = 10000
lowest_loss=None
highest_auc=None

output_list = [net.output, net.loss]

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
                
                for _ in range(semic.num_subj_group[phase]):
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

                        prediction.append(pred.argmax(-1))
                        answer.append(semic.batchset['labels'][i])
                        loss[phase] += cost
                        count[phase] += 1
                        
                prediction = np.concatenate(prediction, axis=0)
                answer = np.concatenate(answer, axis=0)
                
                fpr, tpr, _ = metrics.roc_curve(answer, prediction, pos_label=1)
                tn, fp, fn, tp = metrics.confusion_matrix(answer, prediction).ravel()
                metrics['auc'][phase] = metrics.auc(fpr, tpr)
                metrics['sensitivity'][phase] = tp / (tp+fn)
                metrics['specitivity'][phase] = tn / (tn+fp)
            
            
            summ = tf.Summary()
            for phase in phase_list:
                summ.value.add(tag=phase+'_loss', 
                               simple_value=loss[phase]/count[phase] if count[phase] !=0 else 0)
                for method in metrics.keys():
                    summ.value.add(tag=phase+'_'+method, 
                                   simple_value=metrics[method][phase] if count[phase] !=0 else 0)
                    
            summary_writer.add_summary(summ,epoch)
            
            if epoch > 0 and phase == 'valid':
                if lowest_loss == None or lowest_loss > loss[phase]/count[phase] :
                    lowest_loss = loss[phase]/count[phase]
                    saver.save(sess, ckpt_path+"inception_lowest_loss.ckpt")
                if highest_auc == None or highest_auc < metrics['auc'][phase] :
                    highest_auc = metrics['auc'][phase]
                    saver.save(sess, ckpt_path+"inception_highest_auc.ckpt")
