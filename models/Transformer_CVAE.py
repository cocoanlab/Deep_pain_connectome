# Python 3.6.9
# TF 2.4.0

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import nilearn
import nibabel as nib
from nilearn.image import resample_img, math_img
from nilearn.masking import apply_mask, unmask
from .layers import *
from tensorflow.keras import optimizers as op


def recon_mnispace(dat, schaefer1000_total_mask_idx, roi_mask_order):
    batch_size = dat.shape[0]
    recon = np.zeros((batch_size,)+schaefer1000_total_mask_idx.shape)
    recon_order = roi_mask_order.reshape(-1)
    if type(dat)!=np.ndarray:
        dat = dat.numpy()
        
    for i, pred in enumerate(dat.reshape(batch_size,-1)):
        for j, v in enumerate(pred):
            recon_idx = recon_order[j]
            recon[i,recon_idx] = v
            
    return np.array(recon, np.float32)

class lr_scheduler_config():
    def __init__(self, lr_max, lr_min, lr_T_0, lr_T_mul, num_train_batches, lr_warmup_epoch=0):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_T_0 = lr_T_0
        self.lr_T_mul = lr_T_mul
        self.num_train_batches = num_train_batches
        self.lr_warmup_epoch = lr_warmup_epoch

class VAE(tf.keras.models.Model):
    def __init__(self, data_shape, latent_size=1024, batch_size=None, num_cond = None, 
                 lr_init=1e-3, optimizer_type='adam', l2_reg=0, grad_threshold=None, 
                 only_initialize=None, recon_mask = None, schaefer1000=None,
                 act_type='relu', do_bn=False, KL_beta=1, lr_scheduler_config=None, phase='inference', **kwargs):
            
        super(VAE, self).__init__()
        
        self.__target_affine = np.array([[  -2.,    0.,    0.,   78.],
                                         [   0.,    2.,    0., -112.],
                                         [   0.,    0.,    2.,  -70.],
                                         [   0.,    0.,    0.,    1.]])
        self.__target_shape = (79,95,79)
        
        
        # Loss regularization
        self.__l2_reg = l2_reg
        # Gradient Clipping
        self.__grad_threshold = grad_threshold
        
        self._do_bn = do_bn
        self._act_type = act_type
        
        self.batch_size = batch_size
        self._lr_init = lr_init
        self._optimizer_type = optimizer_type
        self._KL_beta = KL_beta
        
        if recon_mask is None : 
            self.__recon_mask = np.load('../masks/Schaefer2018_1000Parcels_17Networks_order_idx.npy')
            
        if schaefer1000 is None :
            schaefer1000_dat = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=17, resolution_mm=1, resume=True, verbose=1)
            schaefer1000_mask = nib.load(schaefer1000_dat.maps)
            schaefer1000_mask = resample_img(schaefer1000_mask, self.__target_affine, self.__target_shape)
            
            schaefer1000_total_mask = math_img('m>1e-5',m=schaefer1000_mask)
            schaefer1000_total_mask_idx = apply_mask(schaefer1000_mask,schaefer1000_total_mask)
            self.__schaefer1000_total_mask_idx = np.array(schaefer1000_total_mask_idx,np.int)
        
        # model initialization configuration
        if only_initialize is not None:
            only_initialize = only_initialize.lower()            
            # model initialize type check
            if only_initialize is not None and only_initialize not in ['encoder','decoder']:
                raise ValueError("'only_initialize' argument should be declare among 'encoder' or 'decoder', else None to initialize all")
            if only_initialize == 'encoder':
                __init_model_config = [True,False]
            elif only_initialize == 'decoder':
                __init_model_config = [False,True]
        elif only_initialize is None:
            __init_model_config = [True,True]
        
        # phase check. you can use inference phase for valid or test.
        if phase not in ['train', 'inference'] : raise ValueError("phase must be 'train' or 'inference'.")
            
        self._latent_size = latent_size
        self._num_cond = num_cond # Number of Conditions 
        
        #model_input = tf.keras.layers.InputLayer((79,95,79,1), dtype=tf.float32)
        
        if self._num_cond is not None :
            self.encoder_input = self.build_conditional_input('encoder', self._num_cond)
            self.decoder_input = self.build_conditional_input('decoder', self._num_cond)
            data_shape = list(data_shape)
            data_shape[-1] += self._num_cond
        
        self.data_shape = tuple(data_shape) if type(data_shape)!=tuple else data_shape
        
        self.encoder = self.build_encoder(self.data_shape, self._latent_size,linear_key_dim=64, linear_value_dim=64, num_heads=8,)
        self.reparam_latent = self.latent_reparameterization()
        
        if self._num_cond is not None :
            self.decoder = self.build_decoder((latent_size+self._num_cond,), self._latent_size,linear_key_dim=64, linear_value_dim=64, num_heads=8,)
        else :
            self.decoder = self.build_decoder((latent_size,), self._latent_size,linear_key_dim=64, linear_value_dim=64, num_heads=8,act='relu')
        
        self.__set_op(self._lr_init, self._optimizer_type)
        
        # learning rate scheduler - Cosine Annealing
        if lr_scheduler_config:
            self._train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='train_step')
            _config = lr_scheduler_config
            self.lr_scheduler = self.learning_rate_scheduler(lr_init, self.optimizer, _config.lr_max, _config.lr_min,
                                                             _config.lr_T_0, _config.lr_T_mul, _config.num_train_batches, _config.lr_warmup_epoch)

    
    def call(self, input_x, input_cond=None, training=None):
        if input_cond is not None :
            input_dat = self.encoder_input(input_x, input_cond)
        else :
            input_dat = input_x
            
        mu, sigma = self.encoder(input_dat, training)
        latent = self.reparam_latent(mu, sigma)
        
        if input_cond is not None :
            latent = self.decoder_input(latent, input_cond)
        
        recon = self.decoder(latent, training)
        
        #pred = recon_mnispace(recon, self.__schaefer1000_total_mask_idx, self.__recon_mask)
        #true = recon_mnispace(input_x, self.__schaefer1000_total_mask_idx, self.__recon_mask)
        '''
        pix_loss = tf.reduce_mean(tf.keras.losses.MSE(y_pred=recon, y_true=input_x))
        
        corr = self.pearson_correlation(recon, input_x)
        corr = tf.reduce_mean(corr)
        corr = tf.where(tf.math.is_nan(corr), tf.zeros_like(corr), corr)
            '''
        
        corr = self.pearson_correlation(input_x, recon)
        corr = tf.reduce_mean(corr)
        corr = tf.where(tf.math.is_nan(corr), tf.zeros_like(corr), corr)
        pix_loss = tf.reduce_mean(tf.keras.losses.MSE(y_pred=recon,
                                                      y_true=input_x))
        KL = self.kl_divergence(mu, sigma)
#         _thredhold = (1/tf.keras.backend.epsilon()) * self.optimizer.lr
        
#         # KL thresholding for gradient update
#         KL = tf.maximum(KL, 1/_thredhold)
#         KL = tf.minimum(KL, _thredhold)
#         # marginal likelihood thresholoding for gradient update
#         pix_loss = tf.maximum(pix_loss, 1/_thredhold)
#         pix_loss = tf.minimum(pix_loss, _thredhold)
        
        loss = self._KL_beta * KL + pix_loss
        
        return recon, loss, KL, pix_loss, corr
        
    def train(self, input_x, input_cond=None, numpy=False):
        if self._num_cond is not None and input_cond is None:
            raise ValueError('input_cond must be given for conditional vatiational inference.')
            
        with tf.GradientTape() as tape:
            if input_cond is not None:
                pred, loss, KL, pix_loss, corr = self(input_x, input_cond, training=True)
            else :
                pred, loss, KL, pix_loss, corr = self(input_x, training=True)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.__grad_threshold is not None :
            gradients = [tf.clip_by_norm(grad, self.__grad_threshold) for grad in gradients]
        
        if hasattr(self,'lr_scheduler'):
            curr_lr = self.lr_scheduler(self._train_step)
            self._train_step = tf.add(self._train_step, 1)
        else : 
            curr_lr = self.optimizer.lr
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        if numpy :
            return pred.numpy(), loss.numpy(), KL.numpy(), pix_loss.numpy(), corr.numpy(), curr_lr.numpy()
        else : 
            return pred, loss, KL, pix_loss, corr, curr_lr
        
    ######################
    # Model architecture #
    ######################
    
    class build_conditional_input(tf.keras.layers.Layer):
        def __init__(self, model, num_cond, **kwargs):
            super(VAE.build_conditional_input, self).__init__()
            self.__model = model.lower()
            self.__num_cond = num_cond
            assert self.__model in ['encoder','decoder'], 'model should be declared : encoder or decoder'
            
        def call(self,input_x, input_cond):
            
            if self.__model == 'encoder':
                #input_x = tf.keras.layers.InputLayer(VAE.data_shape)
                #input_cond = tf.keras.layers.InputLayer()
                input_cond = tf.squeeze(input_cond)
                one_hot_condition = tf.one_hot(input_cond, depth=self.__num_cond,axis=-1)
                
                cond_input = tf.concat([tf.ones(input_x.shape[:-1]+[1], tf.float32) for _ in range(self.__num_cond)],-1)
                
                for _ in range(1):
                    one_hot_condition = tf.expand_dims(one_hot_condition,1)
                
                cond_input = tf.multiply(cond_input,one_hot_condition)

                input_x = tf.concat((input_x, cond_input),-1)
                return input_x
            
            if self.__model == 'decoder':
                input_cond = tf.squeeze(input_cond)
                one_hot_condition = tf.one_hot(input_cond, depth=self.__num_cond,axis=-1)
                input_x = tf.concat((input_x, one_hot_condition),-1)
                return input_x
            
    # Build Encoder
    class build_encoder(tf.keras.Sequential):
        def __init__(self, input_shape, latent_size, linear_key_dim, linear_value_dim, num_heads, bn=False,
                     act=None, drop_ratio=None, padding='same', use_bias=False):
            
            super(VAE.build_encoder, self).__init__()
            
            model_dim_list = [512,256,128,64,32]
            
            self.add(tf.keras.layers.InputLayer(input_shape))
            for dim_size in model_dim_list:
                self.add(self_attention_layer(linear_key_dim=linear_key_dim, linear_value_dim=linear_value_dim, 
                                              num_heads=num_heads, model_dim=dim_size, use_bias=use_bias))
                
            self.add(tf.keras.layers.Flatten())
            self.add(fc_block(latent_size*2, use_bias=use_bias, bn=bn, act=None))
            
            #self.build(input_shape,)
        
        def call(self, inputs, training=None):
            output = super(VAE.build_encoder, self).call(inputs, training)
            
            mu, sigma = tf.split(output, 2, axis=-1)
            return mu, sigma
    
        
    # Reparameterization
    class latent_reparameterization(tf.keras.layers.Layer):
        def __init__(self):
            super(VAE.latent_reparameterization, self).__init__()
        
        def call(self, mu, sigma, z_sample_mu=.0, z_sample_std=1.):
            eps = tf.random.normal(shape=tf.shape(mu), mean=z_sample_mu, stddev=z_sample_std, dtype=tf.float32)
            
            sigma = tf.exp(sigma * .5)
            sigma = tf.maximum(sigma, tf.keras.backend.epsilon())
            sigma = tf.minimum(sigma, 1/tf.keras.backend.epsilon())
            
            latent = tf.add(mu, tf.multiply(sigma, eps))
            return latent
    
    class build_decoder(tf.keras.Sequential):
        def __init__(self, input_shape, latent_size, linear_key_dim, linear_value_dim, num_heads, bn=False, act=None, 
                     drop_ratio=None, padding='same', use_bias=False):
            
            #assert all([len(last_encode_conv_shape)==4, type(last_encode_conv_shape) is list])
            super(VAE.build_decoder, self).__init__()
            
            model_dim_list = [32,64,128,256,512]
            
            self.add(tf.keras.layers.InputLayer(input_shape))
            self.add(fc_block(1000*32, use_bias=use_bias, bn=bn, act=act))
            self.add(tf.keras.layers.Reshape((1000,32)))
            
            for dim_size in model_dim_list:
                self.add(self_attention_layer(linear_key_dim=linear_key_dim, linear_value_dim=linear_value_dim, 
                                              num_heads=num_heads, model_dim=dim_size, use_bias=use_bias))
            
    class learning_rate_scheduler():
        def __init__(self, lr_init, optimizer, lr_max, lr_min, 
                     lr_T_0, lr_T_mul, num_train_batches, lr_warmup_epoch=0):
            # check necessary arguments
            self.lr_init = lr_init
            self.lr_max = lr_max
            self.lr_min = lr_min
            self.lr_T_0 = lr_T_0
            self.lr_T_mul = lr_T_mul
            self.num_train_batches = num_train_batches
            self.lr_warmup_epoch = lr_warmup_epoch
            self.optimizer = optimizer
            
            self.last_reset = tf.Variable(0, dtype=tf.int32, trainable=False, name='last_reset')
            self.T_i = tf.Variable(self.lr_T_0, dtype=tf.int32, trainable=False, name='T_i')
        
        def __call__(self, train_step):
            train_step = tf.cast(train_step, tf.int32)
            curr_epoch = tf.math.floordiv(train_step, self.num_train_batches)
            T_curr = tf.subtract(curr_epoch,self.last_reset)
            PI = tf.constant(np.pi)
            
            def _do_update():
                update_last_reset = self.last_reset.assign(curr_epoch, use_locking=True)
                update_T_i = self.T_i.assign(tf.multiply(self.T_i, self.lr_T_mul), use_locking=True)
                
                with tf.control_dependencies([update_last_reset, update_T_i]):
                    rate = tf.cast(T_curr, tf.float32) / tf.cast(self.T_i, tf.float32) * PI
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos(rate))
                return lr

            def _no_update():
                rate = tf.cast(T_curr, tf.float32) / tf.cast(self.T_i, tf.float32) * PI
                lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos(rate))
                return lr
            
            learning_rate = tf.cond(tf.greater_equal(T_curr, self.T_i), _do_update, _no_update)
            
            learning_rate = tf.cond(tf.less(curr_epoch, self.lr_warmup_epoch),
                                    lambda: self.lr_init, lambda: learning_rate)
            
            self.optimizer.lr = learning_rate
            return self.optimizer.lr

    def kl_divergence(self, mean, sigma):
        #return tf.reduce_sum(0.5 * ( mean** 2. + s - logvar -1),)
        logvar = tf.exp(sigma)
        logvar = tf.maximum(logvar, tf.keras.backend.epsilon())
        logvar = tf.minimum(logvar, 1/tf.keras.backend.epsilon())
        return tf.reduce_sum(0.5 * ( mean** 2. + logvar - sigma -1),)
    
    def pearson_correlation(self, x, y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        if len(x.shape) != 2:
            x = tf.reshape(x, (tf.shape(x)[0], -1))
        if len(y.shape) != 2:
            y = tf.reshape(y, (tf.shape(y)[0], -1))
        
        x_mu, x_var = [tf.expand_dims(v,-1) for v in tf.nn.moments(x, axes=-1)]
        y_mu, y_var = [tf.expand_dims(v,-1) for v in tf.nn.moments(y, axes=-1)]

        cov_xy = tf.reduce_mean(((x - x_mu)*(y - y_mu)), axis=-1, keepdims=True)
        sigma_xy = tf.sqrt(x_var)*tf.sqrt(y_var)

        return tf.divide(cov_xy, sigma_xy)
    
    def __set_op(self, learning_rate, optimizer_type):
        '''
        Description : 
            set optimizer with given learning rate
            
        Arguments : 
            - learning_rate : float or tensor float. 
            - optimizer_type : [str] {'adam', 'adagrad', 'sgd', 'adadelta'}
            
        Output :
            None
        '''
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type=="adam":
            optimizer = op.Adam(learning_rate)
        elif optimizer_type == "adagrad":
            optimizer = op.Adagrad(learning_rate, initial_accumulator_value=0.0001)
        elif optimizer_type == "sgd":
            optimizer = op.SGD(learning_rate)
        elif optimizer_type == "adadelta":
            optimizer = op.Adadelta(learning_rate,rho=0.95,epsilon=1e-09)
        elif optimizer_type == "radam":
            optimizer = tfa.optimizers.RectifiedAdam(learning_rate)
        else : raise ValueError("{} optimizer doesn't exist.".format(optimizer_type))
        
        self.optimizer = optimizer
        