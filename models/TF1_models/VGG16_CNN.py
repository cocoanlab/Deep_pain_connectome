import numpy as np
import pandas as pd
import tensorflow as tf
from mlxtend.preprocessing import one_hot
import timeit
from random import shuffle
import sys
sys.path.append('..')

from models.layers import *

x = tf.placeholder(tf.float32,shape=[None, 79, 95,1])
y_true = tf.placeholder(tf.float32,shape=[None,2])
is_train = tf.placeholder(tf.bool)
hold_prob = tf.placeholder(tf.float32)
num_classes = 2

class create():
    def __init__(self, data_shape, num_classes, conv_mode='2d', batch_size=None,
                 gpu_memory_fraction=None, optimizer_type='adam', phase='train'):
        
        if phase not in ['train', 'inference'] : raise  ValueError("phase must be 'train' or 'inference'.")
        self.graph = tf.get_default_graph()
        self.num_classes = num_classes
        self.conv_mode = conv_mode
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        
        self.sess = tf.Session(config=config, graph=self.graph)
        
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, (batch_size,) + data_shape, name='input_images')
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_train = tf.placeholder(tf.bool)
            self.__create_model() 
            
            if phase=='train':
                self.y = tf.placeholder(tf.int32, [batch_size,])
                self.lr = tf.placeholder(tf.float32, name="lr")
                
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.output)
                self.loss = tf.reduce_mean(cross_entropy)

                self.sess.run(tf.global_variables_initializer())
                self.train_op = self.__set_op(self.loss, self.lr, optimizer_type)

                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                self.sess.run(tf.variables_initializer(uninit_vars))
                
                
    def __create_model(self):
        convo_1_1 = conv(x, ksize=3, filters=64, ssize=2, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_1_1', 
        bn_name='bn_1_1', bn=True, act=True, is_train=is_train)

        convo_1_2 = conv(convo_1_1, ksize=3, filters=64, ssize=2, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_1_2', 
        bn_name='bn_1_2', bn=True, act=True, is_train=is_train)

        max_pool_1 = max_pooling(convo_1_2, ksize=3, ssize=2, name='max_pool_1')

        convo_2_1 = conv(max_pool_1, ksize=3, filters=128, ssize=1, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_2_1', 
        bn_name='bn_2_1', bn=True, act=True, is_train=is_train)

        convo_2_2 = conv(convo_2_1, ksize=3, filters=128, ssize=1, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_2_2', 
        bn_name='bn_2_2', bn=True, act=True, is_train=is_train)

        max_pool_2 = max_pooling(convo_2_2, ksize=3, ssize=2, name='max_pool_2')

        convo_3_1 = conv(max_pool_2, ksize=3, filters=256, ssize=1, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_3_1', 
        bn_name='bn_3_1', bn=True, act=True, is_train=is_train)

        convo_3_2 = conv(convo_3_1, ksize=3, filters=256, ssize=1, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_3_2', 
        bn_name='bn_3_2', bn=True, act=True, is_train=is_train)

        convo_3_3 = conv(convo_3_2, ksize=3, filters=256, ssize=1, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_3_3', 
        bn_name='bn_3_3', bn=True, act=True, is_train=is_train)

        max_pool_3 = max_pooling(convo_3_3, ksize=3, ssize=2, name='max_pool_3')

        convo_4_1 = conv(max_pool_3, ksize=3, filters=512, ssize=1, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_4_1', 
        bn_name='bn_4_1', bn=True, act=True, is_train=is_train)

        convo_4_2 = conv(convo_4_1, ksize=3, filters=512, ssize=1, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_4_2', 
        bn_name='bn_4_2', bn=True, act=True, is_train=is_train)

        convo_4_3 = conv(convo_4_2, ksize=3, filters=512, ssize=1, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_4_3', 
        bn_name='bn_4_3', bn=True, act=True, is_train=is_train)

        max_pool_4 = max_pooling(convo_4_3, ksize=3, ssize=2, name='max_pool_4')

        convo_5_1 = conv(max_pool_4, ksize=3, filters=512, ssize=1, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_5_1', 
        bn_name='bn_5_1', bn=True, act=True, is_train=is_train)

        convo_5_2 = conv(convo_5_1, ksize=3, filters=512, ssize=1, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_5_2', 
        bn_name='bn_5_2', bn=True, act=True, is_train=is_train)

        convo_5_3 = conv(convo_5_2, ksize=3, filters=512, ssize=1, padding='SAME', 
        use_bias=True, conv_mode='2d' ,conv_name='conv_5_3', 
        bn_name='bn_5_3', bn=True, act=True, is_train=is_train)

        convo_flat = tf.layers.flatten(convo_5_3, name="convo_flat")

        fc1 = fc(convo_flat, 2048, 4096, bn=True, relu=True, is_train=is_train, name='fc1')

        fc1 = dropout(fc1, hold_prob)

        fc2 = fc(fc1, 4096, 4096, bn=True, relu=True, is_train=is_train, name='fc2')

        fc2 = dropout(fc2, hold_prob)

        self.output = fc(fc2, 4096, num_classes, bn=False, relu=False, name='output')

    def __set_op(self, loss_op, learning_rate, optimizer_type="adam"):
        with self.graph.as_default():
            if optimizer_type=="adam":
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif optimizer_type == "adagrad":
                optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.0001)
            elif optimizer_type == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer_type == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
            elif optimizer_type == "adadelta":
                optimizer = tf.train.AdadeltaOptimizer(learning_rate,rho=0.95,epsilon=1e-09)
            else : raise ValueError("{} optimizer doesn't exist.".format(optimizer_type))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss_op)

        return train_op
    
                

