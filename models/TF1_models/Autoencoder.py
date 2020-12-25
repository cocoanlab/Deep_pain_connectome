from models.layers import *
import tensorflow as tf

class create():
    def __init__(self, data_shape, latent_size, batch_size=None, gpu_memory_fraction=None, 
                 optimizer_type='adam', phase='train', disable_decoder=False, enable_skip_connection=False):
        
        if phase not in ['train', 'inference'] : raise  ValueError("phase must be 'train' or 'inference'.")
        self.graph = tf.get_default_graph()
        self.latent_size = latent_size
        self.data_shape = data_shape
        self.disable_decoder = disable_decoder
        self.enable_sc = enable_skip_connection
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        
        self.sess = tf.Session(config=config, graph=self.graph)
        
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, (batch_size,) + (self.data_shape,), name='input')
            self.keep_prob = tf.placeholder(tf.float32, name='dropout_ratio')
            self.is_train = tf.placeholder(tf.bool, name='trainable')
            self.__create_model()
            
            if phase=='train':
                self.lr = tf.placeholder(tf.float32, name="lr")
                self.loss = tf.losses.mean_squared_error(self.x, self.output)
                
                self.sess.run(tf.global_variables_initializer())
                self.train_op = self.__set_op(self.loss, self.lr, optimizer_type)

                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                self.sess.run(tf.variables_initializer(uninit_vars))
                
    def __create_model(self):
        filters = list( map((lambda x : 2**x), reversed( range(7,10)) ))
        encoding_layers = []
        
        for idx, filt_num in enumerate(filters):
            name = 'encode{}'.format(idx+1)
            data = fc(self.x if idx==0 else data, filt_num, bn=True, relu=True, is_train=self.is_train, name=name)
            if self.enable_sc:
                encoding_layers.append(data)
        
        self.latent = fc(data, self.latent_size, bn=True, relu=True, is_train=self.is_train, name='latent')
        
        if not self.disable_decoder :
            filters = list(reversed(filters))
            
            for idx, filt_num in enumerate(filters):
                name = 'decode{}'.format(len(filters)-idx)
                data = fc(self.latent if idx==0 else data, filt_num, bn=True, relu=True, is_train=self.is_train, name=name)
                if self.enable_sc:
                     data = tf.add(data, encoding_layers.pop())
                
            self.output = fc(data, self.data_shape, bn=True, relu=True, is_train=self.is_train, name='output')
        
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