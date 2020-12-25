from models.layers import *
import tensorflow as tf

class create():
    def __init__(self, data_shape, num_output, batch_size=None, gpu_memory_fraction=None,
                 mode='classification', optimizer_type='adam', phase='train'):
        
        if phase not in ['train', 'inference'] : raise  ValueError("phase must be 'train' or 'inference'.")
        self.graph = tf.get_default_graph()
        self.num_output = num_output
        self.data_shape = (data_shape) if type(data_shape)!=tuple else data_shape
        self.mode = mode
        
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
                if self.mode=='classification':
                    self.y = tf.placeholder(tf.int32, [batch_size,], name="ground_truth")
                    self.y_one_hot = tf.one_hot(self.y,depth=self.num_output,axis=-1)
                    self.loss = tf.losses.softmax_cross_entropy(self.y_one_hot, self.logits)
                elif self.mode=='regression':
                    self.y = tf.placeholder(tf.float32, [batch_size,], name="ground_truth")
                    self.loss = tf.losses.mean_squared_error(self.y, tf.squeeze(self.logits))
                else : raise ValueError("Mode should be 'classification' or 'regression'.")
                
                self.sess.run(tf.global_variables_initializer())
                self.lr = tf.placeholder(tf.float32, name="lr")
                self.train_op = self.__set_op(self.loss, self.lr, optimizer_type)

                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                self.sess.run(tf.variables_initializer(uninit_vars))
                
    def __create_model(self):
        PADDING = "VALID"
        USE_BIAS = False 
        
        for idx, filt_size in enumerate([64,128,256,512]):
            data = data if idx != 0 else self.x
            data = batchnorm()
            
            
            data = conv(data, 3, filt_size, ssize=1, padding=PADDING, 
                        conv_mode='3d', use_bias=USE_BIAS, bn=True, act=True, is_train=self.is_train)
            data = conv(data, 3, filt_size, ssize=1, padding=PADDING, 
                        conv_mode='3d', use_bias=USE_BIAS, bn=True, act=True, is_train=self.is_train)
            data = max_pooling(data, ksize=3, ssize=2, mode='3d')
        
        data = tf.layers.flatten(data)
        data = fc(data, 512, bn=True, relu=True, is_train=self.is_train, use_bias=USE_BIAS, name='fc1')
        data = dropout(data, self.keep_prob)
        data = fc(data, 512, bn=True, relu=True, is_train=self.is_train, use_bias=USE_BIAS, name='fc2')
        data = dropout(data, self.keep_prob)
        self.logits = fc(data, self.num_output, bn=False, relu=False, name=self.mode)
        
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