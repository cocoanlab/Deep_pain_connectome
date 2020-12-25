from scripts.CAPS import *
from models import Dense_Resnet
from sklearn import metrics
import tensorflow as tf

dataset_path = list(explore_dir('/data/CAPS/')[1])

train_path = dataset_path[:3]+dataset_path[9:12]
test_path = [dataset_path[3],dataset_path[12]]

tf.reset_default_graph()

net = Dense_Resnet.create((91, 109, 91, 1), 2, conv_mode='3d', optimizer_type='adadelta',)

train_img, train_label = load_dataset(train_path, batch_size=5, shuffle=True)
test_img, test_label = load_dataset(test_path, batch_size=5, shuffle=True)


num_epochs = 100000
lowest_loss=None
highest_auc=None

with net.sess as sess:
    with tf.device('/gpu:0'):
        net.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        summary_writer = tf.summary.FileWriter("./out/", net.sess.graph)
        
        for epoch in range(num_epochs):

            train_auc = 0.
            train_loss = 0.
            train_count = 0
            
            val_auc = 0.
            val_loss = 0.
            val_count = 0
            
            for i in range(len(train_img)):
                pred, cost, op = sess.run([net.output, net.loss,net.train_op], 
                                          feed_dict={net.x: train_img[i],
                                                     net.y: train_label[i], 
                                                     net.lr : 1e-3,
                                                     net.keep_prob: 0.7,
                                                     net.is_train: True})
                pred = pred.argmax(-1)
                fpr, tpr, _ = metrics.roc_curve(train_label[i], pred, pos_label=1)
                train_auc += metrics.auc(fpr, tpr)
                train_loss += cost
                train_count += 1

            for i in range(len(test_img)):
                pred, cost = sess.run([net.output, net.loss], feed_dict={net.x: test_img[i],
                                                                   net.y: test_label[i],
                                                                   net.keep_prob: 1.0,
                                                                   net.is_train: False})
                pred = pred.argmax(-1)
                fpr, tpr, _ = metrics.roc_curve(test_label[i], pred, pos_label=1)
                val_auc += metrics.auc(fpr, tpr)
                val_loss += cost
                val_count += 1
            
            print("[Epochs : "+str(epoch+1)+" ]"+
                  " Train - AUC : {:.5f}".format(train_auc/train_count if train_count !=0 else 0)+
                  " Train - Loss : {:.5f}".format(train_loss/train_count if train_count !=0 else 0)+
                  " Val - AUC : {:.5f}".format(val_auc/val_count if val_count !=0 else 0)+
                  " Val - Loss : {:.5f}".format(val_loss/val_count if val_count !=0 else 0))
            
            summ = tf.Summary()
            summ.value.add(tag='Validation_loss', simple_value=val_loss/val_count if val_count !=0 else 0)
            summ.value.add(tag='Validation_AUC', simple_value=val_auc/val_count if val_count !=0 else 0)
            summ.value.add(tag='Train_loss', simple_value=train_loss/train_count if train_count !=0 else 0)
            summ.value.add(tag='Train AUC', simple_value=train_auc/train_count if train_count !=0 else 0)
            summary_writer.add_summary(summ,epoch)
            
            if epoch > 0:
                if lowest_loss == None or lowest_loss > val_loss/val_count :
                    lowest_loss = val_loss/val_count
                    saver.save(sess, "./out/CAPS/Dense_Resnet_lowest_loss.ckpt")
                if highest_auc == None or highest_auc < val_auc/val_count :
                    highest_auc = val_auc/val_count
                    saver.save(sess, "./out/CAPS/Dense_Resnet_highest_auc.ckpt")
