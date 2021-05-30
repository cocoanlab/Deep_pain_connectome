# Tensorflow 2.4.0

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.keras import backend

class self_attention_layer(Layer):
    def __init__(self, linear_key_dim, linear_value_dim, num_heads, model_dim, use_bias=False, drop_ratio=None):
        super(self_attention_layer, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim

        self._q_proj = tf.keras.layers.Dense(linear_key_dim, use_bias=use_bias,
                                             kernel_initializer=tf.keras.initializers.he_normal(),
                                             bias_initializer=tf.keras.initializers.he_normal())
        self._k_proj = tf.keras.layers.Dense(linear_key_dim, use_bias=use_bias,
                                             kernel_initializer=tf.keras.initializers.he_normal(),
                                             bias_initializer=tf.keras.initializers.he_normal())
        self._v_proj = tf.keras.layers.Dense(linear_value_dim, use_bias=use_bias,
                                             kernel_initializer=tf.keras.initializers.he_normal(),
                                             bias_initializer=tf.keras.initializers.he_normal())
        self.dense1 = tf.keras.layers.Dense(model_dim, use_bias=use_bias,
                                            kernel_initializer=tf.keras.initializers.he_normal(),
                                            bias_initializer=tf.keras.initializers.he_normal())
        self.dense2 = tf.keras.layers.Dense(model_dim, use_bias=use_bias, activation=tf.nn.relu,
                                            kernel_initializer=tf.keras.initializers.he_normal(),
                                            bias_initializer=tf.keras.initializers.he_normal())
        self.dense3 = tf.keras.layers.Dense(model_dim, use_bias=use_bias,
                                            kernel_initializer=tf.keras.initializers.he_normal(),
                                            bias_initializer=tf.keras.initializers.he_normal())
        
        self.LayerNorm = tf.keras.layers.LayerNormalization()
        if drop_ratio is not None:
            self.__do_drop = True
            self._dropout = tf.keras.layers.Dropout(drop_ratio)
        else :
            self.__do_drop = False

    def __split_last_dimension(self, tensor, dim):
        t_shape = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [self.num_heads, dim // self.num_heads])
        return tf.transpose(tensor, [0, 2, 1, 3])

    def call(self, query, key=None, value=None, do_residual=False, training=None):
        if key is None and value is None :
            key = value = query
        q = self._q_proj(query)
        q = self.__split_last_dimension(q, self.linear_key_dim)

        k = self._k_proj(key)
        k = self.__split_last_dimension(k, self.linear_key_dim)

        v = self._v_proj(value)
        v = self.__split_last_dimension(v, self.linear_value_dim)

        key_dim_per_head = self.linear_key_dim // self.num_heads
        output = tf.matmul(q, k, transpose_b=True)
        output = output / (key_dim_per_head**0.5)

        output = tf.nn.softmax(output)
        output = tf.matmul(output, v)

        output = tf.transpose(output, [0, 2, 1, 3])
        t_shape = output.get_shape().as_list()
        num_heads, dim = t_shape[-2:]
        output = tf.reshape(output, [-1] + t_shape[1:-2] + [num_heads * dim])

        output = self.dense1(output)
        if self.__do_drop :
            output = self._dropout(output, training=training)
        if do_residual: output = tf.add(output, value)
        
        output = residual = self.LayerNorm(output)
        
        output = self.dense2(output)
        output = self.dense3(output)
        if self.__do_drop :
            output = self._dropout(output, training=training)
        if do_residual: output = tf.add(output, residual)
        output = self.LayerNorm(output)
        return output

class SeparableConv(Layer):
    def __init__(self, k_size, s_size, num_chn, padding, use_bias, input_dims='2d', bn=False, act=None, xception=False):
        super(SeparableConv, self).__init__()
        
        self.__xception=xception
        
        input_dims = input_dims.upper()
        padding = padding.lower()
        
        self.__do_bn = bn
        assert input_dims in ['2D','3D'], 'Convoltion mode : '+input_dims+' is not available. plz select "2d" or "3d".'        

        self._depthwise_conv = eval(f'Conv{input_dims}')(1, k_size, s_size, padding, use_bias=use_bias,
                                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                                         bias_initializer=tf.keras.initializers.he_normal())
        self._pointwise_conv = eval(f'Conv{input_dims}')(num_chn, 1, 1, padding, use_bias=use_bias,
                                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                                         bias_initializer=tf.keras.initializers.he_normal())
        
        if bn : self._bn = BatchNormalization()
        if act is not None : 
            self.__do_act = True
            self._act = tf.keras.layers.Activation(act)
        else : 
            self.__do_act = False
            
    def call(self,data):
        if self.__xception:
            data = self._pointwise_conv(data)
            data = self._depthwise_conv(data)
        else :
            data = self._depthwise_conv(data)
            data = self._pointwise_conv(data)
        if self.__do_bn : 
            data = self._bn(data)
        if self.__do_act : 
            data = self._act(data)
        return data
    
class SeparableConv_transpose(Layer):
    def __init__(self, k_size, s_size, num_chn, padding, use_bias, input_dims='2d', bn=False, act=None, xception=False):
        super(SeparableConv_transpose, self).__init__()
        
        self.__xception=xception
        
        input_dims = input_dims.upper()
        padding = padding.lower()
        
        self.__do_bn = bn
        assert input_dims in ['2D','3D'], 'Convoltion mode : '+input_dims+' is not available. plz select "2d" or "3d".'        

        self._depthwise_conv = eval(f'Conv{input_dims}Transpose')(1, k_size, s_size, padding, use_bias=use_bias,
                                                                  kernel_initializer=tf.keras.initializers.he_normal(),
                                                                  bias_initializer=tf.keras.initializers.he_normal())
        self._pointwise_conv = eval(f'Conv{input_dims}')(num_chn, 1, 1, padding, use_bias=use_bias,
                                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                                         bias_initializer=tf.keras.initializers.he_normal())
        
        if bn : self._bn = BatchNormalization()
        if act is not None : 
            self.__do_act = True
            self._act = tf.keras.layers.Activation(act)
        else : 
            self.__do_act = False
            
    def call(self,data):
        if self.__xception:
            data = self._pointwise_conv(data)
            data = self._depthwise_conv(data)
        else :
            data = self._depthwise_conv(data)
            data = self._pointwise_conv(data)
        if self.__do_bn : 
            data = self._bn(data)
        if self.__do_act : 
            data = self._act(data)
        return data

class conv_block(Layer):
    def __init__(self, k_size, s_size, num_chn, padding, use_bias, input_dims='2d', bn=False, act=None):
        super(conv_block, self).__init__()
        
        input_dims = input_dims.upper()
        padding = padding.lower()

        self.__do_bn = bn
        assert input_dims in ['2D','3D'], 'Convoltion mode : '+input_dims+' is not available. plz select "2d" or "3d".'

        self._conv = eval(f'Conv{input_dims}')(num_chn, k_size, s_size, padding, use_bias=use_bias,
                                              kernel_initializer=tf.keras.initializers.he_normal(),
                                              bias_initializer=tf.keras.initializers.he_normal())
        if bn : self._bn = BatchNormalization()
        if act is not None : 
            self.__do_act = True
            self._act = tf.keras.layers.Activation(act)
        else : 
            self.__do_act = False

    def call(self, data):
        data = self._conv(data)
        if self.__do_bn : 
            data = self._bn(data)
        if self.__do_act : 
            data = self._act(data)
        return data
    
class residual_conv_block(Layer):
    def __init__(self, k_size, s_size, num_chn, padding, use_bias, input_dims='2d', bn=False, act=None):
        super(residual_conv_block, self).__init__()
        
        input_dims = input_dims.upper()
        padding = padding.lower()
        self.__do_bn = bn
        assert input_dims in ['2D','3D'], 'Convoltion mode : '+input_dims+' is not available. plz select "2d" or "3d".'
        
        self._conv1 = conv_block(k_size, 1 if s_size != 1 else s_size, num_chn, padding, use_bias=use_bias, input_dims=input_dims, bn=bn, act=act)
        self._conv2 = conv_block(k_size, s_size, num_chn, padding, use_bias=use_bias, input_dims=input_dims, bn=bn, act=None)
        self._conv3 = conv_block(1, s_size, num_chn, padding, use_bias=use_bias, input_dims=input_dims, bn=bn, act=None)
        
        if act is not None : 
            self.__do_act = True
            self._act = tf.keras.layers.Activation(act)
        else : 
            self.__do_act = False
    
    def call(self, data):
        data1 = self._conv1(data)
        data1 = self._conv2(data1)
        
        data2 = self._conv3(data)
        
        data = data1+data2
        
        if self.__do_act : 
            data = self._act(data)
        return data
    
class deconv_block(Layer):
    def __init__(self, k_size, s_size, num_chn, padding, use_bias, input_dims='2d', bn=False, act=None):
        super(deconv_block, self).__init__()
        input_dims = input_dims.upper()
        padding = padding.lower()

        self.__do_bn = bn

        assert input_dims in ['2D','3D'], 'Convoltion input dimension : ['+input_dims+'] is not available. plz select "2d" or "3d".'

        self._conv = eval(f'Conv{input_dims}Transpose')(num_chn, k_size, s_size, padding, use_bias=use_bias,
                                                        kernel_initializer=tf.keras.initializers.he_normal(),
                                                        bias_initializer=tf.keras.initializers.he_normal())
        if bn : self._bn = BatchNormalization()
        if act is not None : 
            self.__do_act = True
            self._act = tf.keras.layers.Activation(act)
        else : 
            self.__do_act = False

    def call(self, data):
        data = self._conv(data)
        if self.__do_bn : 
            data = self._bn(data)
        if self.__do_act : 
            data = self._act(data)
        return data
    
class fc_block(Layer):
    def __init__(self, num_out, use_bias=False, drop_ratio=None, bn=False, act=None):
        super(fc_block, self).__init__()

        self.__do_bn = bn

        self._fc = Dense(num_out, use_bias=use_bias,
                         kernel_initializer=tf.keras.initializers.he_normal(),
                         bias_initializer=tf.keras.initializers.he_normal())
        if bn : self._bn = BatchNormalization()
        if act is not None : 
            self.__do_act = True
            self._act = tf.keras.layers.Activation(act)
        else : 
            self.__do_act = False
            
        if drop_ratio is not None:
            self.__do_drop = True
            self._dropout = tf.keras.layers.Dropout(drop_ratio)
        else :
            self.__do_drop = False
            
    def call(self, data, training=None):
        data = self._fc(data)
        if self.__do_drop :
            data = self._dropout(data, training=training)
        if self.__do_bn : 
            data = self._bn(data)
        if self.__do_act : 
            data = self._act(data)
        return data

class max_pool(Layer):
    def __init__(self, k_size=3, s_size=2, padding='same', input_dims='2d'):
        super(max_pool, self).__init__()
        padding = padding.lower()
        input_dims = input_dims.upper()
        assert input_dims in ['2D','3D'], 'Pooling input dimension : ['+input_dims+'] is not available. plz select "2d" or "3d".'
        self._pool = eval(f'MaxPooling{input_dims}')(k_size, s_size, padding)

    def call(self, data):
        return self._pool(data)

class avg_pool(Layer):
    def __init__(self, avg_type='local', k_size=3, s_size=2, padding='same', input_dims='2d'):
        super(avg_pool, self).__init__()
        padding = padding.lower()
        input_dims = input_dims.upper()
        avg_type = avg_type.lower()
        assert input_dims in ['2D','3D'], 'Pooling input dimension : ['+input_dims+'] is not available. plz select "2d" or "3d".'
        assert avg_type in ['local','global'], 'average pooling type : ['+avg_type+'] is not available. plz select "local" or "global".'

        if avg_type == 'local':
            self._pool = eval(f'AveragePooling{input_dims}')(k_size, s_size, padding)
        elif avg_type == 'global':
            self._pool = eval(f'GlobalAveragePooling{input_dims}')()

    def call(self, data):
        return self._pool(data)

class squeeze_excitation(Layer):
    def __init__(self, ratio, input_dims='2d'):
        super(squeeze_excitation, self).__init__()
        input_dims = input_dims.upper()
        assert input_dims in ['2D','3D'], 'Pooling input dimension : ['+input_dims+'] is not available. plz select "2d" or "3d".'

        orig_chn = data.get_shape().as_list()[-1]

        self._gap = eval(f'GlobalAveragePooling{input_dims}')()
        self._squeeze = Dense(orig_chn/ratio, use_bias=use_bias, activation='relu')
        self._excite = Dense(orig_chn, use_bias=use_bias, activation='sigmoid')

        self.__dimension = (-1,1,1,orig_chn) if input_dims == '2D' else (-1,1,1,1,orig_chn)

    def call(self, data):
        output = self._gap(data)
        output = self._squeeze(output)
        output = self._excite(output)
        output = tf.reshape(output, self.__dimension)
        output = tf.math.multiply(data,output)

        return output