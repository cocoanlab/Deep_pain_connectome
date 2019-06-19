from models.layers import *
import tensorflow as tf
import numpy as np 

def Stem(data, bn=True, act=True, use_bias=False, is_train=None):
    
    with tf.variable_scope('Inception_v4-Stem'):
        data = conv(data, 3, 32, ssize=2, padding="VALID", 
                    conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        data = conv(data, 3, 32, ssize=1, padding="VALID", 
                    conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        data = conv(data, 3, 64, ssize=1, padding="SAME", 
                    conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        data_down_A1 = conv(data, 3, 64, ssize=2, padding="VALID", 
                            conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        data_down_A2 = conv(data, 3, 96, ssize=2, padding="VALID", 
                            conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        data = tf.concat((data_down_A1, data_down_A2),axis=-1)
        
        data_1 = conv(data, 1, 64, ssize=1, padding="SAME", 
                      conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        data_1 = conv(data_1, 3, 96, ssize=1, padding="VALID", 
                      conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        
        data_2 = conv(data, 1, 64, ssize=1, padding="SAME", 
                      conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        data_2 = conv(data_2, (7,1,1), 64, ssize=1, padding="SAME", 
                      conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        data_2 = conv(data_2, (1,7,1), 64, ssize=1, padding="SAME", 
                      conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        data_2 = conv(data_2, (1,1,7), 64, ssize=1, padding="SAME", 
                      conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        data_2 = conv(data_2, 3, 96, ssize=1, padding="VALID", 
                      conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)

        data = tf.concat((data_1, data_2),axis=-1)
        data_down_B1 = conv(data, 3, 192, ssize=2, padding="VALID", 
                            conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
        data_down_B2 = conv(data, 3, 192, ssize=2, padding="VALID", 
                            conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
    return tf.concat((data_down_B1, data_down_B2),axis=-1)
    
    
def inception_residul_block(data, block_type, idx, bn=True, act=True, use_bias=False, is_train=None):
    with tf.variable_scope('Inception_v4-Residual_Block'+'-'+block_type.upper()+str(idx+1)):
        data = tf.nn.relu(data)
        
        if block_type.upper() == 'A':
            data_1 = conv(data, 1, 32, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)            
            
            data_2 = conv(data, 1, 32, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            data_2 = conv(data_2, 3, 32, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            
            data_3 = conv(data, 1, 32, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            data_3 = conv(data_3, 3, 48, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            data_3 = conv(data_3, 3, 64, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            
            concat_data = tf.concat((data_1, data_2, data_3),axis=-1)
            concat_data = conv(concat_data, 1, int(data.shape[-1]), ssize=1, padding="SAME", 
                               conv_mode='3d', use_bias=False, bn=bn,act=act, is_train=is_train)
            concat_data = tf.keras.activations.linear(concat_data)
                
            data = tf.add(data, concat_data)
            return tf.nn.relu(data)
                
        elif block_type.upper() in ['B', 'C']:
            if block_type.upper() == 'B':
                kernels = [(7,1,1),(1,7,1),(1,1,7)]
                filters = [128,160,192,244]
            elif block_type.upper() == 'C':
                kernels = [(3,1,1),(1,3,1),(1,1,3)]
                filters = [192,224,256,288]
            
            data_1 = conv(data, 1, filters[-1 if block_type.upper() == 'B' else 0], ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)            
            
            data_2 = conv(data, 1, filters[0], ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
            data_2 = conv(data_2, kernels[0], filters[1], ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
            data_2 = conv(data_2, kernels[1], filters[2], ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
            data_2 = conv(data_2, kernels[2], filters[3], ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias,bn=bn,act=act, is_train=is_train)
            
            concat_data = tf.concat((data_1, data_2),axis=-1)
            concat_data = conv(concat_data, 1, int(data.shape[-1]), ssize=1, padding="SAME", 
                               conv_mode='3d', use_bias=False, bn=bn,act=act, is_train=is_train)
            concat_data = tf.keras.activations.linear(concat_data)
            
            data = tf.add(data, concat_data)
            return tf.nn.relu(data)
        
        else : 
            raise ValueError("Inception Residual Block Type must be in 'A', 'B', 'C'.")
        
def inception_reduction_block(data, block_type, bn=True, act=True, use_bias=False, is_train=None): 
    with tf.variable_scope('Inception_v4-Reduction_Block'+'-'+block_type.upper()):
        
        if block_type.upper() == 'A':

            k, l, m, n, =  256, 256, 384, 384
            
            data_1 = conv(data, 1, k, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            data_1 = conv(data_1, 3, l, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            data_1 = conv(data_1, 3, m, ssize=2, padding="VALID", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            
            data_2 = conv(data, 3, n, ssize=2, padding="VALID", 
                          conv_mode='3d', use_bias=use_bias, bn=bn, act=act, is_train=is_train)
            
            data_3 = conv(data, 3, int(data.shape[-1]), ssize=2, padding="VALID", 
                          conv_mode='3d', use_bias=use_bias, bn=bn, act=act, is_train=is_train)
            
            return tf.concat((data_1, data_2, data_3),axis=-1)
                               
            
        elif block_type.upper() == 'B':
            data_1 = conv(data, 1, 256, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            data_1 = conv(data_1, 3, 288, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            data_1 = conv(data_1, 3, 320, ssize=2, padding="VALID", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            
            data_2 = conv(data, 1, 256, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            data_2 = conv(data_2, 3, 288, ssize=2, padding="VALID", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            
            data_3 = conv(data, 1, 256, ssize=1, padding="SAME", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            data_3 = conv(data_3, 3, 384, ssize=2, padding="VALID", 
                          conv_mode='3d', use_bias=use_bias, bn=bn,act=act, is_train=is_train)
            
            data_4 = conv(data, 3, int(data.shape[-1]), ssize=2, padding="VALID", 
                          conv_mode='3d', use_bias=use_bias, bn=bn, act=act, is_train=is_train)
            
            return tf.concat((data_1, data_2, data_3, data_4),axis=-1)
        
        else : 
            raise ValueError("Inception Reduction Block Type must be 'A' or 'B'.")

def squeeze_excitation_layer(data, num_out, ratio, stage, idx=None):
    layer_name = 'Squeeze_Excitation'+'-'+stage.upper()
    layer_name += str(idx+1) if idx != None else ''
    
    with tf.variable_scope(layer_name):
        squeeze = global_avg_pooling(data, '3d', name='GAP')

        excitation = fc(squeeze, num_out/ratio, bn=False, relu=True, name='fc1')
        excitation = fc(excitation, num_out, bn=False, relu=False, name='fc2')
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,1,num_out])

        scale = data * excitation

        return scale
            
class create:
    def __init__(self, data_shape, num_output, mode='classification', batch_size=None, gpu_memory_fraction=None, optimizer_type='adam', phase='train', reduction_ratio=4):
        
        if phase not in ['train', 'inference'] : raise  ValueError("phase must be 'train' or 'inference'.")
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.num_output = num_output
        self.reduction_ratio = reduction_ratio
        self.mode = mode.lower()
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, (batch_size,) + data_shape, name='input')
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_train = tf.placeholder(tf.bool)
            self.__create_model() 
            self.sess = tf.Session(config=config, graph=self.graph)
            self.logits = tf.nn.softmax(self.output, name='logits')
            
            if phase=='train':
                self.y = tf.placeholder(tf.int32, [batch_size,], name="ground_truth")
                self.lr = tf.placeholder(tf.float32, name="learning_rate")
                
                if self.mode=='classification':
                    self.y_one_hot = tf.one_hot(self.y,depth=2,axis=-1)
                    self.loss = tf.losses.softmax_cross_entropy(self.y_one_hot, self.logits)
                elif self.mode=='regression':
                    self.loss = tf.losses.mean_squared_error(self.y, tf.squeeze(self.logits))
                else : raise ValueError("Mode should be 'classification' or 'regression'.")
                
                self.sess.run(tf.global_variables_initializer())
                self.train_op = self.__set_op(self.loss, self.lr, optimizer_type)
                
                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                
                self.sess.run(tf.variables_initializer(uninit_vars))
            
    def __create_model(self):

        fmri = Stem(self.x, is_train=self.is_train)
        fmri = squeeze_excitation_layer(fmri, int(fmri.shape[-1]), self.reduction_ratio, 'Stem')
        
        for idx in range(5):
            fmri = inception_residul_block(fmri, 'A', idx, is_train=self.is_train)
            fmri = squeeze_excitation_layer(fmri, int(fmri.shape[-1]), self.reduction_ratio, 'A', idx)
            
        fmri = inception_reduction_block(fmri, 'A', is_train=self.is_train) 
        fmri = squeeze_excitation_layer(fmri, int(fmri.shape[-1]), self.reduction_ratio, 'Reduction_A')
        
        for idx in range(10):
            fmri = inception_residul_block(fmri, 'B', idx, is_train=self.is_train)
            fmri = squeeze_excitation_layer(fmri, int(fmri.shape[-1]), self.reduction_ratio, 'B', idx)
        
        fmri = inception_reduction_block(fmri, 'B', is_train=self.is_train) 
        fmri = squeeze_excitation_layer(fmri, int(fmri.shape[-1]), self.reduction_ratio, 'Reduction_B')
        
        for idx in range(5):
            fmri = inception_residul_block(fmri, 'C', idx, is_train=self.is_train)
            fmri = squeeze_excitation_layer(fmri, int(fmri.shape[-1]), self.reduction_ratio, 'C', idx)

        fmri = global_avg_pooling(fmri, '3d', name='GAP')
        
        fmri = tf.layers.flatten(fmri)
        fmri = dropout(fmri, self.keep_prob)
        self.output = fc(fmri, self.num_output, bn=False, relu=False, name=self.mode)
        self.logits = tf.nn.softmax(self.output, name='logits')
        
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
