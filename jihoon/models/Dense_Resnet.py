from models.layers import *
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np

def residual_block(data, kernel_size, filters, stage, block, conv_mode='2d', bn=False, act=False, use_bias=True, is_train=None):
    with tf.variable_scope('Residual_Block_stage'+str(stage)+'-'+block):
        suffix=str(stage)+block+"_branch"
        conv_name_base = "res"+suffix
        bn_name_base = "bn"+suffix

        conv1 = conv(data, kernel_size, filters[0], ssize=1, padding="SAME",conv_name=conv_name_base+"2a",
                     conv_mode=conv_mode, bn_name=bn_name_base+"2a",use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        conv2 = conv(conv1, kernel_size, filters[1], ssize=1, padding="SAME",conv_name=conv_name_base+"2b",
                     conv_mode=conv_mode, bn_name=bn_name_base+"2b",use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        conv3 = conv(conv2, kernel_size, filters[2], ssize=1, padding="SAME",conv_name=conv_name_base+"2c",
                     conv_mode=conv_mode, bn_name=bn_name_base+"2c",use_bias=use_bias,bn=bn,act=False, is_train=is_train)

        if int(data.shape[-1])!=filters[2]:
            shortcut = conv(data, 1, filters[2], ssize=1, padding="SAME", conv_mode=conv_mode, 
                            conv_name=conv_name_base+"shortcut", use_bias=False, bn=False, act=False)
        else :
            shortcut = data
        addx_h = tf.add(conv3, shortcut, name="res"+str(stage)+block+"_shortcut_sum")
        return tf.nn.relu(addx_h, name="res"+str(stage)+block+"_out")

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
        padding_type='SAME'
        BIAS = False
        
        conv_down1 = conv(self.x,filters=64,ksize=9,ssize=4,padding="SAME",use_bias=BIAS, conv_mode=self.conv_mode, 
                          conv_name="conv_down1", bn_name="bn_down1",bn=True,act=True, is_train=self.is_train)
        
        # Stage 2
        stage2_filters = [64,64,256]
        resblock_1 = residual_block(conv_down1,7,stage2_filters, stage=2, block="a",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_2 = residual_block(resblock_1, 7, stage2_filters, stage=2, block="b",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_3 = residual_block(resblock_2, 7, [64,64,128], stage=2, block="c",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        conv_down2 = conv(resblock_3,filters=128,ksize=7,ssize=2,padding="SAME",use_bias=BIAS, conv_mode=self.conv_mode, 
                          conv_name="conv_down2", bn_name="bn_down2",bn=True,act=True, is_train=self.is_train)
        conv_down1_dense1 = conv(conv_down1,filters=128,ksize=2,ssize=2,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down1_dense1", bn_name="bn_down1_dense1",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        stage2_dense = tf.concat((conv_down2, conv_down1_dense1),
                                 axis=-1, name='stage2_dense')
        
        # Stage 3
        stage3_filters = [128,128,512]
        resblock_4 = residual_block(stage2_dense ,5,stage3_filters, stage=3, block="a", 
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_5 = residual_block(resblock_4, 5, stage3_filters, stage=3, block="b",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_6 = residual_block(resblock_5, 5, stage3_filters, stage=3, block="c",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_7 = residual_block(resblock_6, 5, [128,128,256], stage=3, block="d",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        conv_down3 = conv(resblock_7,filters=256, ksize=5,ssize=2,padding="SAME",use_bias=False, conv_mode=self.conv_mode, 
                          conv_name="conv_down3",bn_name="bn_down3",bn=True,act=True, is_train=self.is_train)
        conv_down2_dense1 = conv(stage2_dense,filters=128,ksize=2,ssize=2,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down2_dense1", bn_name="bn_down2_dense1",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        conv_down1_dense2 = conv(conv_down1,filters=128,ksize=4,ssize=4,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down1_dense2", bn_name="bn_down1_dense2",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        stage3_dense = tf.concat((conv_down3, conv_down2_dense1,conv_down1_dense2),
                                 axis=-1, name='stage3_dense')
        
        # Stage 4
        stage4_filters = [256,256,1024]
        stage4 = residual_block(stage3_dense,3,stage4_filters, stage=4, block="a",
                                conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        for i in range(5):
            stage4 = residual_block(stage4, 3, stage4_filters, stage=4, block=chr(ord('b')+i),
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        stage4 = residual_block(stage4, 3, [256,256,512], stage=4, block=chr(ord('b')+i+1),
                                conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
            
        conv_down4 = conv(stage4,filters=256,ksize=2,ssize=2,padding="SAME",use_bias=False, conv_mode=self.conv_mode, 
                          conv_name="conv_down4",bn_name="bn_down4",bn=True,act=True, is_train=self.is_train)
        conv_down3_dense1 = conv(stage3_dense,filters=256,ksize=2,ssize=2,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down3_dense1", bn_name="bn_down3_dense1",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        conv_down2_dense2 = conv(stage2_dense,filters=256,ksize=4,ssize=4,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down2_dense2", bn_name="bn_down2_dense2",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        conv_down1_dense3 = conv(conv_down1,filters=256,ksize=8,ssize=8,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down1_dense3", bn_name="bn_down1_dense3",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        stage4_dense = tf.concat((conv_down4, conv_down3_dense1,conv_down2_dense2,conv_down1_dense3),
                                 axis=-1, name='stage4_dense')
        # Stage 5
        stage5_filters = [512,512,2048]
        resblock_14 = residual_block(stage4_dense,3,stage5_filters, stage=5, block="a",
                                     conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_15 = residual_block(resblock_14, 3, stage5_filters, stage=5, block="b",
                                     conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_16 = residual_block(resblock_15, 3, [512,512,1024], stage=5, block="c",
                                     conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        
        conv_down5 = conv(resblock_16,filters=512, ksize=2,ssize=2,padding="SAME",use_bias=False, conv_mode=self.conv_mode, 
                          conv_name="conv_down5",bn_name="bn_down5",bn=True,act=True, is_train=self.is_train)
        conv_down4_dense1 = conv(stage4_dense,filters=512,ksize=2,ssize=2,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down4_dense1", bn_name="bn_down4_dense1",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)        
        conv_down3_dense2 = conv(stage3_dense,filters=512,ksize=4,ssize=4,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down3_dense2", bn_name="bn_down3_dense2",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        conv_down2_dense3 = conv(stage2_dense,filters=512,ksize=8,ssize=8,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down2_dense3", bn_name="bn_down2_dense3",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        conv_down1_dense4 = conv(conv_down1,filters=512,ksize=16,ssize=16,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down1_dense4", bn_name="bn_down1_dense4",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        stage5_dense = tf.concat((conv_down5, conv_down4_dense1, conv_down3_dense2,
                                  conv_down2_dense3, conv_down1_dense4),
                                 axis=-1, name='stage5_dense')
        num_nodes=1
        for i in range(1,len(stage5_dense.shape)): num_nodes*=int(stage5_dense.shape[i])
        stage5_dense = tf.reshape(stage5_dense, [-1, num_nodes])
        
        fc1 = fc(stage5_dense, num_nodes, 4096, bn=True, relu=True, is_train=self.is_train, name='fc1')
        fc1 = dropout(fc1, self.keep_prob)
        fc2 = fc(fc1, 4096, 4096, bn=True, relu=True, is_train=self.is_train, name='fc2')
        fc2 = dropout(fc1, self.keep_prob)
        
        self.output = fc(fc2, 4096, self.num_classes, bn=False, relu=False, name='output')


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
