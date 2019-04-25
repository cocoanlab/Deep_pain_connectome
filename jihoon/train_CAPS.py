from scripts.CAPS import *
from models import inception_v4
from sklearn import metrics

import tensorflow as tf

dataset_path = list(explore_dir('/data/CAPS/')[1])

train_path = dataset_path[:3]+dataset_path[9:12]
test_path = [dataset_path[3],dataset_path[12]]

train_img, train_label = load_dataset(train_path, batch_size=20, shuffle=True)
test_img, test_label = load_dataset(test_path, batch_size=20, shuffle=True)

train_answer = list(np.concatenate(train_label, axis=-1))
test_answer = list(np.concatenate(test_label, axis=-1))

net = inception_v4.create(data_shape=(91, 109, 91, 1), num_output=2, mode='classification',optimizer_type='adadelta', phase='train')

num_epochs = 100000
lowest_loss=None
highest_auc=None

with net.sess as sess:
    with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        summary_writer = tf.summary.FileWriter("./out/", net.sess.graph)
        
        for epoch in range(num_epochs):

            train_pred = []
            train_loss = 0.
            train_count = 0
            
            val_pred = []
            val_loss = 0.
            val_count = 0
            
            for i in range(len(train_img)):
                pred, cost, _ = sess.run([net.output, net.loss,net.train_op], 
                                          feed_dict={net.x: train_img[i],
                                                     net.y: train_label[i], 
                                                     net.lr : 1e-3,
                                                     net.keep_prob: 0.7,
                                                     net.is_train: True})
                train_pred += list(pred.argmax(-1))
                train_loss += cost
                train_count += 1

            for i in range(len(test_img)):
                pred, cost = sess.run([net.output, net.loss], feed_dict={net.x: test_img[i],
                                                                   net.y: test_label[i],
                                                                   net.keep_prob: 1.0,
                                                                   net.is_train: False})
                val_pred += list(pred.argmax(-1))
                val_loss += cost
                val_count += 1
            
            # Train - AUC, Sensitivity, Specitivity
            fpr, tpr, _ = metrics.roc_curve(train_answer, train_pred, pos_label=1)
            tn, fp, fn, tp = metrics.confusion_matrix(train_answer, train_pred).ravel()
            
            train_auc = metrics.auc(fpr, tpr)
            train_sensitivity = tp / (tp+fn)
            train_specitivity = tn / (tn+fp)
            
            # Valid - AUC, Sensitivity, Specitivity
            fpr, tpr, _ = metrics.roc_curve(test_answer, val_pred, pos_label=1)
            tn, fp, fn, tp = metrics.confusion_matrix(test_answer, val_pred).ravel()
            
            valid_auc = metrics.auc(fpr, tpr)
            valid_sensitivity = tp / (tp+fn)
            valid_specitivity = tn / (tn+fp)
            
            print("[Epochs : "+str(epoch+1)+" ]"+
                  " Train - AUC : "+str(train_auc)+
                  " Train - Sensitivity : "+str(train_sensitivity)+
                  " Train - Specitivity : "+str(train_specitivity)+
                  " Train - Loss : "+str(train_loss/train_count if train_count !=0 else 0)+
                  " Val - AUC : "+str(valid_auc)+
                  " Val - Sensitivity : "+str(valid_sensitivity)+
                  " Val - Specitivity : "+str(valid_specitivity)+
                  " Val - Loss : "+str(val_loss/val_count if val_count !=0 else 0))
            
            summ = tf.Summary()
            summ.value.add(tag='Validation_loss', simple_value=val_loss/val_count if val_count !=0 else 0)
            summ.value.add(tag='Validation_AUC', simple_value=valid_auc)
            summ.value.add(tag='Validation_Sensitivity', simple_value=valid_sensitivity)
            summ.value.add(tag='Validation_Specitivity', simple_value=valid_specitivity)
            
            summ.value.add(tag='Train_loss', simple_value=train_loss/train_count if train_count !=0 else 0)
            summ.value.add(tag='Train_AUC', simple_value=train_auc)
            summ.value.add(tag='Train_Sensitivity', simple_value=train_sensitivity)
            summ.value.add(tag='Train_Specitivity', simple_value=train_specitivity)
            summary_writer.add_summary(summ,epoch)
            
            if epoch > 0:
                if lowest_loss == None or lowest_loss > val_loss/val_count :
                    lowest_loss = val_loss/val_count
                    saver.save(sess, "./out/CAPS/inception_lowest_loss.ckpt")
                if highest_auc == None or highest_auc < valid_auc :
                    highest_auc = valid_auc
                    saver.save(sess, "./out/CAPS/inception_highest_auc.ckpt")