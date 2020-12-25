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

def squeeze_excitation_layer(data, num_out, ratio, stage, idx=None, mode='2d'):
    layer_name = 'Squeeze_Excitation_stage'+str(stage)
    layer_name += '-'+idx if idx != None else '-Reduction'
    
    with tf.variable_scope(layer_name):
        squeeze = global_avg_pooling(data, mode, name='GAP_'+mode.upper())

        excitation = fc(squeeze, num_out/ratio, bn=False, relu=True, name='fc1')
        excitation = fc(excitation, num_out, bn=False, relu=False, name='fc2')
        excitation = tf.nn.sigmoid(excitation)
        
        output_size = [-1,1,1,num_out] if mode == '2d' else [-1,1,1,1,num_out]
        excitation = tf.reshape(excitation, output_size)

        scale = data * excitation

        return scale
    
class create():
    def __init__(self, data_shape, num_outputs, reduction_ratio, conv_mode='2d', batch_size=None, 
                 gpu_memory_fraction=None, optimizer_type='adam', phase='train', task='classification'):
        
        if phase not in ['train', 'inference'] : raise  ValueError("phase must be 'train' or 'inference'.")
        self.graph = tf.get_default_graph()
        self.conv_mode = conv_mode
        self.reduction_ratio = reduction_ratio
        self.task = task.lower()
        self.num_outputs = num_outputs
        
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
                self.lr = tf.placeholder(tf.float32, name="lr")
                
                if self.task == 'classification':
                    self.y = tf.placeholder(tf.int32, [batch_size,])
                    self.y_one_hot = tf.one_hot(self.y,depth=self.num_outputs ,axis=-1)
                    self.loss = tf.losses.softmax_cross_entropy(self.y_one_hot, self.logits)
                elif self.task == 'regression':
                    self.y = tf.placeholder(tf.float32, [batch_size,])
                    self.loss = tf.losses.mean_squared_error(self.y, tf.squeeze(self.logits))
                elif self.task == 'both' :
                    self.y_cls = tf.placeholder(tf.int32, [batch_size,])
                    self.y_reg = tf.placeholder(tf.float32, [batch_size,])

                    self.y_one_hot = tf.one_hot(self.y_cls, depth=6, axis=-1)
                    self.loss_cls = tf.losses.softmax_cross_entropy(self.y_one_hot, self.logits_cls)
                    self.loss_reg = tf.losses.mean_squared_error(self.y_reg, tf.squeeze(self.logits_reg))
                    
                    self.total_loss = tf.multiply(self.loss_cls, self.loss_reg)
                    self.total_loss = tf.add(self.total_loss, self.loss_cls)
                    self.total_loss = tf.add(self.total_loss, self.loss_reg)
                    
                else : 
                    raise ValueError("Task should be 'classification', 'regression' or 'both'.")

                self.sess.run(tf.global_variables_initializer())
                self.train_op = self.__set_op(self.total_loss, self.lr, optimizer_type)

                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                self.sess.run(tf.variables_initializer(uninit_vars))

    def __create_model(self):
        padding_type='SAME'
        BIAS = False
        
        conv_down1 = conv(self.x,filters=32,ksize=9,ssize=4,padding="SAME",use_bias=BIAS, conv_mode=self.conv_mode, 
                          conv_name="conv_down1", bn_name="bn_down1",bn=True,act=True, is_train=self.is_train)
        conv_down1 = squeeze_excitation_layer(conv_down1, int(conv_down1.shape[-1]),
                                              self.reduction_ratio, '1', mode=self.conv_mode)
        
        # Stage 2
        stage2_filters = [32,32,128]
        resblock_1 = residual_block(conv_down1,7,stage2_filters, stage=2, block="a",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_1 = squeeze_excitation_layer(resblock_1, int(resblock_1.shape[-1]), 
                                              self.reduction_ratio, '2', 'a', mode=self.conv_mode)
        
        resblock_2 = residual_block(resblock_1, 7, stage2_filters, stage=2, block="b",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_2 = squeeze_excitation_layer(resblock_2, int(resblock_2.shape[-1]), 
                                              self.reduction_ratio, '2', 'b', mode=self.conv_mode)
        
        resblock_3 = residual_block(resblock_2, 7, [32,32,64], stage=2, block="c",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_3 = squeeze_excitation_layer(resblock_3, int(resblock_3.shape[-1]), 
                                              self.reduction_ratio, '2', 'c', mode=self.conv_mode)
        
        conv_down2 = conv(resblock_3,filters=64,ksize=7,ssize=2,padding="SAME",use_bias=BIAS, conv_mode=self.conv_mode, 
                          conv_name="conv_down2", bn_name="bn_down2",bn=True,act=True, is_train=self.is_train)
        conv_down1_dense1 = conv(conv_down1,filters=64,ksize=2,ssize=2,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down1_dense1", bn_name="bn_down1_dense1",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        stage2_dense = tf.concat((conv_down2, conv_down1_dense1),
                                 axis=-1, name='stage2_dense')
        stage2_dense = squeeze_excitation_layer(stage2_dense, int(stage2_dense.shape[-1]),
                                                self.reduction_ratio, '2', mode=self.conv_mode)
        
        # Stage 3
        stage3_filters = [64,64,256]
        resblock_4 = residual_block(stage2_dense ,5,stage3_filters, stage=3, block="a", 
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_4 = squeeze_excitation_layer(resblock_4, int(resblock_4.shape[-1]), 
                                              self.reduction_ratio, '3', 'a', mode=self.conv_mode)
        
        resblock_5 = residual_block(resblock_4, 5, stage3_filters, stage=3, block="b",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_5 = squeeze_excitation_layer(resblock_5, int(resblock_5.shape[-1]), 
                                              self.reduction_ratio, '3', 'b', mode=self.conv_mode)
        
        resblock_6 = residual_block(resblock_5, 5, stage3_filters, stage=3, block="c",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_6 = squeeze_excitation_layer(resblock_6, int(resblock_6.shape[-1]), 
                                              self.reduction_ratio, '3', 'c', mode=self.conv_mode)
        
        resblock_7 = residual_block(resblock_6, 5, [64,64,128], stage=3, block="d",
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_7 = squeeze_excitation_layer(resblock_7, int(resblock_7.shape[-1]), 
                                              self.reduction_ratio, '3', 'd', mode=self.conv_mode)
        
        conv_down3 = conv(resblock_7,filters=128, ksize=5,ssize=2,padding="SAME",use_bias=False, conv_mode=self.conv_mode, 
                          conv_name="conv_down3",bn_name="bn_down3",bn=True,act=True, is_train=self.is_train)
        conv_down2_dense1 = conv(stage2_dense,filters=64,ksize=2,ssize=2,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down2_dense1", bn_name="bn_down2_dense1",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        conv_down1_dense2 = conv(conv_down1,filters=64,ksize=4,ssize=4,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down1_dense2", bn_name="bn_down1_dense2",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        stage3_dense = tf.concat((conv_down3, conv_down2_dense1,conv_down1_dense2),
                                 axis=-1, name='stage3_dense')
        stage3_dense = squeeze_excitation_layer(stage3_dense, int(stage3_dense.shape[-1]),
                                                self.reduction_ratio, '3', mode=self.conv_mode)
        
        # Stage 4
        stage4_filters = [128,128,512]
        stage4 = residual_block(stage3_dense,3,stage4_filters, stage=4, block="a",
                                conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        stage4 = squeeze_excitation_layer(stage4, int(stage4.shape[-1]), 
                                          self.reduction_ratio, '4', 'a', mode=self.conv_mode)
        
        for i in range(5):
            stage4 = residual_block(stage4, 3, stage4_filters, stage=4, block=chr(ord('b')+i),
                                    conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
            stage4 = squeeze_excitation_layer(stage4, int(stage4.shape[-1]), 
                                              self.reduction_ratio, '4', chr(ord('b')+i), mode=self.conv_mode)
        stage4 = residual_block(stage4, 3, [128,128,256], stage=4, block=chr(ord('b')+i+1),
                                conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        stage4 = squeeze_excitation_layer(stage4, int(stage4.shape[-1]), 
                                          self.reduction_ratio, '4', chr(ord('b')+i+1), mode=self.conv_mode)
        
        conv_down4 = conv(stage4,filters=128,ksize=2,ssize=2,padding="SAME",use_bias=False, conv_mode=self.conv_mode, 
                          conv_name="conv_down4",bn_name="bn_down4",bn=True,act=True, is_train=self.is_train)
        conv_down3_dense1 = conv(stage3_dense,filters=128,ksize=2,ssize=2,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down3_dense1", bn_name="bn_down3_dense1",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        conv_down2_dense2 = conv(stage2_dense,filters=128,ksize=4,ssize=4,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down2_dense2", bn_name="bn_down2_dense2",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        conv_down1_dense3 = conv(conv_down1,filters=128,ksize=8,ssize=8,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down1_dense3", bn_name="bn_down1_dense3",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        stage4_dense = tf.concat((conv_down4, conv_down3_dense1,conv_down2_dense2,conv_down1_dense3),
                                 axis=-1, name='stage4_dense')
        stage4_dense = squeeze_excitation_layer(stage4_dense, int(stage4_dense.shape[-1]),
                                                self.reduction_ratio, '4', mode=self.conv_mode)
        
        # Stage 5
        stage5_filters = [256,256,1024]
        resblock_14 = residual_block(stage4_dense,3,stage5_filters, stage=5, block="a",
                                     conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_14 = squeeze_excitation_layer(resblock_14, int(resblock_14.shape[-1]), 
                                              self.reduction_ratio, '5', 'a', mode=self.conv_mode)
        
        resblock_15 = residual_block(resblock_14, 3, stage5_filters, stage=5, block="b",
                                     conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_15 = squeeze_excitation_layer(resblock_15, int(resblock_15.shape[-1]), 
                                              self.reduction_ratio, '5', 'b', mode=self.conv_mode)
        
        resblock_16 = residual_block(resblock_15, 3, [256,256,512], stage=5, block="c",
                                     conv_mode=self.conv_mode, is_train=self.is_train, use_bias=BIAS)
        resblock_16 = squeeze_excitation_layer(resblock_16, int(resblock_16.shape[-1]), 
                                              self.reduction_ratio, '5', 'c', mode=self.conv_mode)
        
        conv_down5 = conv(resblock_16,filters=256, ksize=2,ssize=2,padding="SAME",use_bias=False, conv_mode=self.conv_mode, 
                          conv_name="conv_down5",bn_name="bn_down5",bn=True,act=True, is_train=self.is_train)
        conv_down4_dense1 = conv(stage4_dense,filters=256,ksize=2,ssize=2,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down4_dense1", bn_name="bn_down4_dense1",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)        
        conv_down3_dense2 = conv(stage3_dense,filters=256,ksize=4,ssize=4,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down3_dense2", bn_name="bn_down3_dense2",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        conv_down2_dense3 = conv(stage2_dense,filters=256,ksize=8,ssize=8,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down2_dense3", bn_name="bn_down2_dense3",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        conv_down1_dense4 = conv(conv_down1,filters=256,ksize=16,ssize=16,padding="SAME",use_bias=BIAS,
                                 conv_name="conv_down1_dense4", bn_name="bn_down1_dense4",
                                 conv_mode=self.conv_mode, bn=True,act=True, is_train=self.is_train)
        stage5_dense = tf.concat((conv_down5, conv_down4_dense1, conv_down3_dense2,
                                  conv_down2_dense3, conv_down1_dense4),
                                 axis=-1, name='stage5_dense')
        stage5_dense = squeeze_excitation_layer(stage5_dense, int(stage5_dense.shape[-1]),
                                                self.reduction_ratio, '5', mode=self.conv_mode)
        
        self.stage5_dense = stage5_dense
        stage5_dense = tf.layers.flatten(stage5_dense)
        
        fc1 = fc(stage5_dense, 4096, bn=True, relu=True, is_train=self.is_train, name='fc1')
        fc2 = fc(fc1, 4096, bn=True, relu=True, is_train=self.is_train, name='fc2')
        fc3 = fc(fc2, 4096, bn=True, relu=True, is_train=self.is_train, name='fc3')
        
        if self.task != 'both':
            self.output = fc(fc3, self.outputs, bn=False, relu=False, name=self.mode)
            self.logits = tf.nn.softmax(self.output, name='logits')
        else :
            self.output_cls = fc(fc3, 6, bn=False, relu=False, name='classification')
            self.output_reg = fc(fc3, 1, bn=False, relu=False, name='regression')
            self.logits_cls = tf.nn.softmax(self.output_cls, name='logits')
            self.logits_reg = tf.nn.softmax(self.output_reg, name='logits')

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
