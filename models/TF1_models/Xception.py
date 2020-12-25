from models.layers import *
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np

USE_FUSED_BN = True
BN_EPSILON = 0.001
BN_MOMENTUM = 0.99

def reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])]
    return kernel_size_out

def separable_conv(data, depth, is_train, name, act=True, bn=True, data_format='channels_last'):
    with tf.variable_scope(name):
        bn_axis = -1 if data_format == 'channels_last' else 1

        if act : data = tf.nn.relu(data, name='active_ReLU')
            
        data = tf.layers.separable_conv3d(data, depth, (3, 3),
                            strides = (1, 1), padding='SAME',
                            data_format = data_format,
                            activation = None , use_bias = False,
                            depthwise_initializer = tf.contrib.layers.xavier_initializer(),
                            pointwise_initializer = tf.contrib.layers.xavier_initializer(),
                            bias_initializer = tf.zeros_initializer(),
                            name='Separable_Convolution', reuse=None)
        if bn :
            data = tf.layers.batch_normalization(data, momentum=BN_MOMENTUM, name= 'Batch_Norm', axis=bn_axis,
                                    epsilon=BN_EPSILON, training=is_train, reuse=None, fused=USE_FUSED_BN)
        return data
        
def squeeze_excitation_layer(data, num_out, ratio, name):
    with tf.variable_scope(name):
        squeeze = global_avg_pooling(data)

        excitation = fc(squeeze, num_out/ratio, bn=False, relu=True, name='fc1')
        excitation = fc(excitation, num_out, bn=False, relu=False, name='fc2')
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,num_out])

        scale = data * excitation

        return scale
        
class create():
    def __init__(self, data_shape, num_output, reduction_ratio, weight_decay, data_format='channels_last', batch_size=None, 
                 enable_SE = False, gpu_memory_fraction=None, optimizer_type='adam', phase='train'):
        
        if phase not in ['train', 'inference'] : raise  ValueError("phase must be 'train' or 'inference'.")
        self.graph = tf.get_default_graph()
        self.num_output = num_output
        self.reduction_ratio = reduction_ratio
        self.weight_decay = weight_decay
        self.enable_SE = enable_SE
        self.data_format = data_format
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        
        self.sess = tf.Session(config=config, graph=self.graph)
        
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, (batch_size,) + data_shape, name='input_images')
            self.is_train = tf.placeholder(tf.bool)
            self.__create_model() 
            
            if phase=='train':
                self.y = tf.placeholder(tf.int32, [batch_size,])
                self.y_one_hot = tf.one_hot(self.y,depth=2,axis=-1)
                self.lr = tf.placeholder(tf.float32, name="lr")
                
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_one_hot, logits=self.logits))
                l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
                self.loss = cost + l2_loss * weight_decay

                self.sess.run(tf.global_variables_initializer())
                self.train_op = self.__set_op(self.loss, self.lr, optimizer_type)

                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                self.sess.run(tf.variables_initializer(uninit_vars))

    def __create_model(self):
        # [ Entry Flow ] - Stage1
        data = conv(self.x, 3, 32, 2, padding="VALID", conv_name="block1-conv", bn_name="block1-bn", 
                    conv_mode='3d', use_bias=False, bn=True, act=True, is_train=self.is_train)
        data = conv(data, 3, 64, 1, padding="VALID", conv_name="block2-conv", bn_name="block2-bn",
                    conv_mode='3d', use_bias=False, bn=True, act=True, is_train=self.is_train)
        residual = conv(data, 1, 128, 2, padding="VALID", conv_name="block3-res_conv", bn_name="block3-res_bn",
                    conv_mode='3d', use_bias=False, bn=True, act=False, is_train=self.is_train)
        
        # [ Entry Flow ] - Stage2
        data = separable_conv(data, 128, self.is_train, act=False, name='block4-dws_conv')
        data = separable_conv(data, 128, self.is_train, name='block5-dws_conv')
        data = max_pooling(data)
        data = tf.add(data, residual)
        if self.enable_SE : data = squeeze_excitation_layer(data, 128, self.reduction_ratio, 'SE1')
        residual = conv(data, 1, 256, 2, padding="VALID", conv_name="block6-res_conv", bn_name="block6_res_bn",
                    conv_mode='3d', use_bias=False, bn=True, act=False, is_train=self.is_train)
        
        # [ Entry Flow ] - Stage3
        data = separable_conv(data, 256, self.is_train, name='block7-dws_conv')
        data = separable_conv(data, 256, self.is_train, name='block8-dws_conv')
        data = max_pooling(data)
        data = tf.add(data, residual)
        if self.enable_SE : data = squeeze_excitation_layer(data, 256, self.reduction_ratio, 'SE2')
        residual = conv(data, 1, 728, 2, padding="VALID", conv_name="block9-res_conv", bn_name="block9_res_bn",
                    conv_mode='3d', use_bias=False, bn=True, act=False, is_train=self.is_train)
        
        # [ Entry Flow ] - Stage 4
        data = separable_conv(data, 728, self.is_train, name='block10-dws_conv')
        data = separable_conv(data, 728, self.is_train, name='block11-dws_conv')
        data = max_pooling(data)
        data = tf.add(data, residual)
        if self.enable_SE : data = squeeze_excitation_layer(data, 728, self.reduction_ratio, 'SE3')
        
        # [ Middle Flow ] - Stage 5
        block_num = 12
        SE_num = 4
        for _ in range(8):
            residual = data
            
            for _ in range(3):
                data = separable_conv(data, 728, self.is_train, name='block{}-dws_conv'.format(block_num))
                block_num+=1
            data = tf.add(data, residual)
            if self.enable_SE : 
                data = squeeze_excitation_layer(data, 728, self.reduction_ratio, 'SE{}'.format(SE_num))
                SE_num+=1
            
        # [ Exit Flow ] - Stage 6
        residual = conv(data, 1, 1024, 2, conv_name="block37-res_conv", bn_name="block37_res_bn",
                    conv_mode='3d', use_bias=False, bn=True, act=False, is_train=self.is_train, padding="SAME")
        data = separable_conv(data, 728, self.is_train, name='block38-dws_conv')
        data = separable_conv(data, 1024, self.is_train, name='block39-dws_conv')
        data = max_pooling(data)
        data = tf.add(data, residual)
        if self.enable_SE : data = squeeze_excitation_layer(data, 1024, self.reduction_ratio, 'SE13')
        
        # [ Exit Flow ] - Stage 7
        data = separable_conv(data, 1536, self.is_train, act=False, name='block40-dws_conv')
        data = separable_conv(data, 2048, self.is_train, name='block41-dws_conv')
        data = tf.nn.relu(data)
        if self.enable_SE : data = squeeze_excitation_layer(data, 2048, self.reduction_ratio, 'SE14')

        if self.data_format == 'channels_first':
            channels_last_inputs = tf.transpose(data, [0, 2, 3, 1])
        else:
            channels_last_inputs = data
        
        reduced = reduced_kernel_size_for_small_input(channels_last_inputs, [10, 10])
        data = tf.layers.average_pooling2d(data, pool_size = reduced, strides = 1, 
                                           padding='valid', data_format=self.data_format, name='avg_pool')

        if self.data_format == 'channels_first':
            data = tf.squeeze(data, axis=[2, 3])
        else:
            data = tf.squeeze(data, axis=[1, 2])
            
        self.logits = fc(data, self.num_output, name='FC', relu=False, bn=False, is_train=self.is_train)
        
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