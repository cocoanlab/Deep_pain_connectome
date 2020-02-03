from models.layers import *
import tensorflow as tf

class create():
    def __init__(self, data_shape, latent_size, batch_size=None, gpu_memory_fraction=None, enable_sc=False,
                 optimizer_type='adam', phase='train', plain='total', disable_decoder=False):
        
        if phase not in ['train', 'inference'] : raise  ValueError("phase must be 'train' or 'inference'.")
        if plain not in ['total', 'axial', 'sagittal'] : raise  ValueError("phase must be 'total', 'axial', 'saggital'.")
        self.graph = tf.get_default_graph()
        self.latent_size = latent_size
        self.data_shape = data_shape
        self.disable_decoder = disable_decoder
        self.plain = plain
        self.enable_sc = enable_sc
        self.__dimension = '3d' if plain == 'total' else '2d'
        self.__cropping = {
            'total' : [
                ((0, 0), (0, 0), (0, 1)),
                ((0, 0), (0, 0), (0, 0)),
                ((1, 0), (1, 0), (0, 0)),
            ],
            'sagittal' : [
                ((0, 0), (1, 0)),
                ((1, 0), (0, 0)), 
                ((1, 0), (1, 0)),
            ],
            'axial' : [
                ((1, 0), (0, 0)),
                ((0, 0), (1, 0)), 
                ((1, 0), (1, 0)),
            ]
        }
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        
        self.sess = tf.Session(config=config, graph=self.graph)
        
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, (batch_size,) + self.data_shape, name='input')
            self.keep_prob = tf.placeholder(tf.float32, name='dropout_ratio')
            self.is_train = tf.placeholder(tf.bool, name='trainable')
            self.__create_model()
            
            if phase=='train':
                self.lr = tf.placeholder(tf.float32, name="lr")
                self.loss = tf.losses.mean_squared_error(tf.layers.flatten(self.x), 
                                                         tf.layers.flatten(self.output))
                
                self.sess.run(tf.global_variables_initializer())
                self.train_op = self.__set_op(self.loss, self.lr, optimizer_type)

                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                self.sess.run(tf.variables_initializer(uninit_vars))
                
    def __create_model(self):
        encoding_layers = []
        
        data = conv(self.x, 3, 32, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = conv(data, 3, 32, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        encoding_layers.append(data)
        data = max_pooling(data, ksize=3, ssize=2, mode=self.__dimension)
        
        data = conv(data, 3, 64, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = conv(data, 3, 64, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        encoding_layers.append(data)
        data = max_pooling(data, ksize=3, ssize=2, mode=self.__dimension)

        data = conv(data, 3, 128, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = conv(data, 3, 128, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        encoding_layers.append(data)
        data = max_pooling(data, ksize=3, ssize=2, mode=self.__dimension)

        data = conv(data, 3, 256, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = conv(data, 3, 256, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        encoding_layers.append(data)
        data = max_pooling(data, ksize=3, ssize=2, mode=self.__dimension)
        
        orig_shape = tuple(map(int,data.get_shape()[1:]))
        data = tf.layers.flatten(data)
        self.latent = fc(data, self.latent_size, bn=False, relu=False, is_train=self.is_train, name='latent')

        if not self.disable_decoder :
            data = fc(self.latent, data.get_shape()[-1], bn=True, relu=True, is_train=self.is_train, name='fc_reconstruction')
            data = tf.reshape(data, (-1,)+orig_shape)
            
            data = deconv(data, 3, 256, ssize=2, padding="SAME", deconv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
            if self.__dimension == '2d' :
                data = tf.keras.layers.Cropping2D(cropping=self.__cropping[self.plain][0])(data)
            elif self.__dimension == '3d':
                data = tf.keras.layers.Cropping3D(cropping=self.__cropping[self.plain][0])(data)
            if self.enable_sc:
                data = tf.add(data, encoding_layers.pop())
            data = conv(data, 3, 256, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
            data = conv(data, 3, 256, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)

            data = deconv(data, 3, 128, ssize=2, padding="SAME", deconv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
            if self.__dimension == '2d' :
                data = tf.keras.layers.Cropping2D(cropping=self.__cropping[self.plain][0])(data)
            elif self.__dimension == '3d':
                data = tf.keras.layers.Cropping3D(cropping=self.__cropping[self.plain][0])(data)
            if self.enable_sc:
                data = tf.add(data, encoding_layers.pop())
            data = conv(data, 3, 128, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
            data = conv(data, 3, 128, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)

            data = deconv(data, 3, 64, ssize=2, padding="SAME", deconv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
            if self.__dimension == '2d' :
                data = tf.keras.layers.Cropping2D(cropping=self.__cropping[self.plain][1])(data)
            elif self.__dimension == '3d':
                data = tf.keras.layers.Cropping3D(cropping=self.__cropping[self.plain][1])(data)
            if self.enable_sc:
                data = tf.add(data, encoding_layers.pop())
            data = conv(data, 3, 64, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
            data = conv(data, 3, 64, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)

            data = deconv(data, 3, 32, ssize=2, padding="SAME", deconv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
            if self.__dimension == '2d' :
                data = tf.keras.layers.Cropping2D(cropping=self.__cropping[self.plain][2])(data)
            elif self.__dimension == '3d':
                data = tf.keras.layers.Cropping3D(cropping=self.__cropping[self.plain][2])(data)
            if self.enable_sc:
                data = tf.add(data, encoding_layers.pop())
            data = conv(data, 3, 32, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
            data = conv(data, 3, 32, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
            self.output = conv(data, 3, 1, ssize=1, padding="SAME", conv_mode=self.__dimension, use_bias=False, bn=False,act=False, is_train=self.is_train)
        
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