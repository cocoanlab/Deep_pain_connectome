from models.layers import *
import tensorflow as tf

class lr_scheduler_config():
    def __init__(self, lr_max, lr_min, lr_T_0, lr_T_mul, 
                 batch_size, num_train_batches,):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_T_0 = lr_T_0
        self.lr_T_mul = lr_T_mul
        self.num_train_batches = num_train_batches

class create():
    def __init__(self, data_shape, latent_size=1024, batch_size=None, num_cond = None,
                 gpu_memory_fraction=None,  lr_init=1e-3, optimizer_type='adam', 
                 l2_reg=0, grad_threshold=None,
                 KL_beta=None, phase='train', lr_scheduler_config=None):
        
        # Loss regularization
        self.__l2_reg=l2_reg
        self.__grad_threshold=grad_threshold
        
        
        if phase not in ['train', 'inference'] : raise  ValueError("phase must be 'train' or 'inference'.")
        self.graph = tf.get_default_graph()
        self.latent_size = latent_size
        self.num_cond = num_cond
        self.data_shape = (data_shape) if type(data_shape)!=tuple else data_shape
        
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
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
            if self.num_cond is not None :
                self.cond = tf.placeholder(tf.int32, [batch_size,], name="condition_number")
            self.__create_model()
            
            if phase=='train':
                #self.lr = tf.placeholder(tf.float32, name="lr")
                #self.loss = tf.losses.mean_squared_error(self.y, tf.squeeze(self.logits))
                with tf.name_scope('loss') : 
                    orig_x = tf.layers.flatten(self.x)
                    recon_x = tf.layers.flatten(self.output)
                    
                    marginal_likelihood = tf.reduce_sum(orig_x * tf.log(recon_x) + (1 - orig_x) * tf.log(1 - recon_x), 1)
                    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
                    
                    KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(self.sigma) - tf.log(1e-8 + tf.square(self.sigma)) - 1, 1)
                    KL_divergence = tf.reduce_mean(KL_divergence) # * KL_beta
                    '''
                    self.KL_divergence = KL_divergence
                    ELBO = marginal_likelihood - KL_divergence
                    self.loss = -ELBO
                    '''
                    self.KL_divergence = KL_divergence
                    self.recon_loss = tf.losses.mean_squared_error(orig_x,recon_x)
                    self.loss = self.KL_divergence+self.recon_loss
                
                self.sess.run(tf.global_variables_initializer())
                
                if lr_scheduler_config is not None:
                    self.lr = self.schedule_lr(lr_init, lr_scheduler_config)
                else : 
                    self.lr = lr_init
                self.train_op = self.__set_op(self.loss, self.lr, optimizer_type)

                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                self.sess.run(tf.variables_initializer(uninit_vars))
                
    def schedule_lr(self, lr, config, name='learning_rate_scheduler'):
        lr_max = config.lr_max
        lr_min = config.lr_min
        lr_T_0 = config.lr_T_0
        lr_T_mul = config.lr_T_mul
        num_train_batches = config.num_train_batches
        
        with tf.name_scope(name) :
            curr_epoch = self.global_step // num_train_batches
            last_reset = tf.Variable(0, dtype=tf.int32, trainable=False, name='last_reset')

            T_i = tf.Variable(lr_T_0, dtype=tf.int32, trainable=False, name='T_i')
            T_curr = curr_epoch - last_reset
        
            def _do_update():
                update_last_reset = tf.assign(last_reset, curr_epoch, use_locking=True)
                update_T_i = tf.assign(T_i, T_i * lr_T_mul, use_locking=True)
                with tf.control_dependencies([update_last_reset, update_T_i]):
                    rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
                    lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
                return lr

            def _no_update():
                rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
                lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
                return lr
        
            learning_rate = tf.cond(tf.greater_equal(T_curr, T_i), _do_update, _no_update)
            
        return learning_rate
    
    def __create_model(self):
        PADDING = "SAME"
        USE_BIAS = True 
        filter_list = [64,128,256]
        
        with tf.name_scope('Gaussian_Encoder'):
            for idx, filt_size in enumerate(filter_list):
                #trainable = False if self.__freeze_conv else True
                trainable = True
                if idx == 0:
                    data = self.x
                    if self.num_cond is not None :
                        self.one_hot_condition = one_hot_condition = tf.one_hot(self.cond,depth=self.num_cond,axis=-1)
                        for _ in range(3):
                            one_hot_condition = tf.expand_dims(one_hot_condition,1)
                        
                        cond_input = tf.concat([tf.ones_like(self.x, tf.float32) for _ in range(self.num_cond)],-1)
                        cond_input*=one_hot_condition
                        data = tf.concat((self.x, cond_input),-1)
                    
                data = conv(data, 3, filt_size, ssize=1, padding=PADDING, trainable=trainable,
                            conv_mode='3d', use_bias=USE_BIAS, bn=True, act=True, is_train=self.is_train)
                data = conv(data, 3, filt_size, ssize=1, padding=PADDING, trainable=trainable,
                            conv_mode='3d', use_bias=USE_BIAS, bn=True, act=True, is_train=self.is_train)
                
                if idx != len(filter_list) -1 :
                    data = max_pooling(data, ksize=3, ssize=2, mode='3d')
                
            last_conv_shape = data.shape[1:].as_list()
            data = tf.layers.flatten(data)
            flatten_size = data.shape[1:].as_list()[0]
            #data = global_avg_pooling(data, mode='3d', name='GAP_flatten')
            data = fc(data, self.latent_size*2, bn=True, relu=False, is_train=self.is_train, name='latent')
            
            self.mu = data[:, :self.latent_size]
            self.sigma = 1e-6 + tf.nn.softplus(data[:, self.latent_size:])
            
        with tf.name_scope('latent'):
            self.latent = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
            self.latent = tf.where(tf.math.is_nan(self.latent), tf.zeros_like(self.latent), self.latent)
            self.latent = tf.where(tf.math.is_inf(self.latent), tf.zeros_like(self.latent), self.latent)
            if self.num_cond is not None :
                self.latent = tf.concat((self.latent, self.one_hot_condition),-1)
        
        with tf.name_scope('Gaussian_Decoder'):
            data = fc(self.latent, flatten_size, bn=True, relu=True, is_train=self.is_train, name='recon_flatten')
            data = tf.reshape(data, [-1]+last_conv_shape)
            
            reversed_filter_size = list(reversed(filter_list))
            for idx, filt_size in enumerate(reversed_filter_size):
                #trainable = False if self.__freeze_conv else True
                trainable = True
                data = conv(data, 3, filt_size, ssize=1, padding=PADDING, trainable=trainable,
                            conv_mode='3d', use_bias=USE_BIAS, bn=True, act=True, is_train=self.is_train)
                data = conv(data, 3, filt_size, ssize=1, padding=PADDING, trainable=trainable,
                            conv_mode='3d', use_bias=USE_BIAS, bn=True, act=True, is_train=self.is_train)
                if idx != len(filter_list) -1 :
                    recon_filt_size = reversed_filter_size[idx+1]
                    data = deconv(data, 3, recon_filt_size, ssize=2, padding=PADDING, 
                                  use_bias=USE_BIAS, deconv_mode='3d' ,bn=True, act=True, is_train=self.is_train)
                else :
                    data = conv(data, 1, 1, ssize=1, padding=PADDING, trainable=trainable,
                                conv_mode='3d', use_bias=USE_BIAS, bn=True, act=False, is_train=self.is_train)
            self.output = data[:,1:,1:,1:]
        
    def __set_op(self, loss_op, learning_rate, optimizer_type="adam"):
        l2_reg = self.__l2_reg
        grad_threshold = self.__grad_threshold
        
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
            elif optimizer_type == 'radam' : 
                optimizer = RAdamOptimizer(learning_rate=learning_rate)
            else : raise ValueError("{} optimizer doesn't exist.".format(optimizer_type))
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            vars_to_train = tf.trainable_variables()
                
            if l2_reg > 0 :
                l2_losses = [tf.reduce_sum(var**2) for var in vars_to_train]
                l2_losses = tf.add_n(l2_losses)
                loss_op += l2_reg * l2_losses
                
            # Clip gradient at given threshold
            if grad_threshold is not None :
                assert grad_threshold is not None, 'Need "grad_threshold" to clip gradients'
                clipped = []
                gradients = tf.gradients(loss_op, vars_to_train)
                for grad in gradients:
                    if isinstance(gradients, tf.IndexedSlices):
                        c_grad = tf.clip_by_norm(grad.values, grad_threshold)
                        c_grad = tf.IndexedSlices(grad.indices, c_grad)
                    else :
                        c_g = tf.clip_by_norm(grad, grad_threshold)

                    clipped.append(grad)
                gradients = clipped
                
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss_op, global_step=self.global_step, var_list=vars_to_train)

            self.sess.run(tf.variables_initializer(optimizer.variables()))
                
        return train_op