from models.layers import *
import tensorflow as tf

class create():
    def __init__(self, data_shape, num_output, batch_size=None, gpu_memory_fraction=None, optimizer_type='adam', phase='train'):
        
        if phase not in ['train', 'inference'] : raise  ValueError("phase must be 'train' or 'inference'.")
        self.graph = tf.get_default_graph()
        self.data_shape = data_shape
        self.num_output = num_output
        self.__dimension = '3d'
        
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
                self.y = tf.placeholder(tf.int32, [batch_size,], name="ground_truth")
                self.lr = tf.placeholder(tf.float32, name="learning_rate")
                
                self.y_one_hot = tf.one_hot(self.y,depth=self.num_output,axis=-1)
                self.loss = tf.losses.softmax_cross_entropy(self.y_one_hot, self.logits)
                
                self.sess.run(tf.global_variables_initializer())
                self.train_op = self.__set_op(self.loss, self.lr, optimizer_type)
                
                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                
                self.sess.run(tf.variables_initializer(uninit_vars))
                
    def __create_model(self):
        data = conv(self.x, 3, 32, ssize=1, padding="VALID", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = conv(data, 3, 32, ssize=1, padding="VALID", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = max_pooling(data, ksize=3, ssize=2, mode=self.__dimension)
            
        data = conv(data, 3, 64, ssize=1, padding="VALID", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = conv(data, 3, 64, ssize=1, padding="VALID", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = max_pooling(data, ksize=3, ssize=2, mode=self.__dimension)
        
        data = conv(data, 3, 128, ssize=1, padding="VALID", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = conv(data, 3, 128, ssize=1, padding="VALID", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = max_pooling(data, ksize=3, ssize=2, mode=self.__dimension)
        
        data = conv(data, 3, 256, ssize=1, padding="VALID", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = conv(data, 3, 256, ssize=1, padding="VALID", conv_mode=self.__dimension, use_bias=False, bn=True,act=True, is_train=self.is_train)
        data = max_pooling(data, ksize=3, ssize=2, mode=self.__dimension)
        
        #data = tf.layers.flatten(data)
        data = global_avg_pooling(data, mode='3d')
        data = fc(data, 4096, bn=True, relu=True, name='fc1', is_train=self.is_train)
        data = dropout(data, self.keep_prob)
        data = fc(data, 4096, bn=True, relu=True, name='fc2', is_train=self.is_train)
        data = dropout(data, self.keep_prob)
        self.output = fc(data, self.num_output, bn=False, relu=False, name='output')
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