import tensorflow as tf

def conv(data, ksize, filters, ssize, padding, use_bias, conv_mode='2d' ,conv_name=None, bn_name=None, bn=False, act=True, is_train=None):
    if bn and is_train==None:
        raise ValueError('Phase should be declared "True" for train or "False" for test when you activate batch normalization.')
    
    if conv_mode.lower() == '2d':
        output = tf.layers.conv2d(data, kernel_size=ksize, filters=filters,
                                  strides=(ssize,ssize),
                                  padding=padding.upper(),
                                  name=conv_name,use_bias=use_bias,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
    elif conv_mode.lower() == '3d':
        output = tf.layers.conv3d(data, kernel_size=ksize, filters=filters,
                                  strides=(ssize,ssize,ssize),
                                  padding=padding.upper(),
                                  name=conv_name,use_bias=use_bias,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
    else :
        raise ValueError('Convoltion mode : ['+conv_mode+'] is not available. plz select "2d" or "3d".')
    
    if bn : output = batch_norm(output, is_train, bn_name)
    if act : output = tf.nn.relu(output)
    return output

def deconv(data, ksize, filters, ssize, padding, use_bias, deconv_mode='2d' ,deconv_name=None, bn_name=None, bn=False, act=True, is_train=None):
    if bn and is_train==None:
        raise ValueError('Phase should be declared "True" for train or "False" for test when you activate batch normalization.')
    
    if deconv_mode.lower() == '2d':
        output = tf.layers.conv2d_transpose(data, kernel_size=ksize, filters=filters,
                                            strides=(ssize,ssize),
                                            padding=padding.upper(),
                                            name=deconv_name,use_bias=use_bias,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    elif deconv_mode.lower() == '3d':
        output = tf.layers.conv3d_transpose(data, kernel_size=ksize, filters=filters,
                                            strides=(ssize,ssize,ssize),
                                            padding=padding.upper(),
                                            name=deconv_name,use_bias=use_bias,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    else :
        raise ValueError('Convoltion mode : ['+conv_mode+'] is not available. plz select "2d" or "3d".')
    
    if bn : output = batch_norm(output, is_train, bn_name)
    if act : output = tf.nn.relu(output)
    return output

def max_pooling(data, ksize=3, ssize=2, mode='2d', name=None):
    with tf.name_scope(name):
        if mode == '2d':
            return tf.nn.max_pool(data, ksize=[1,ksize,ksize,1], strides=[1,ssize,ssize,1], padding="SAME", name=name)
        elif mode == '3d':
            return tf.nn.max_pool3d(data, ksize=[1,ksize,ksize,ksize,1], strides=[1,ssize,ssize,ssize,1], padding="SAME", name=name)
        else :
            raise ValueError('Max Pooling mode : ['+conv_mode+'] is not available. plz select "2d" or "3d".')

def avg_pooling(data, ksize=3, ssize=2, mode='2d', name=None):
    with tf.name_scope(name):
        if mode == '2d' : 
            return tf.nn.avg_pool(data, ksize=[1,ksize,ksize,1], strides=[1,ssize,ssize,1], padding="VALID", name=name)
        elif mode == '3d' :
            return tf.nn.avg_pool3d(data, ksize=[1,ksize,ksize,ksize,1], strides=[1,ssize, ssize,ssize,1], padding="VALID", name=name)

def global_avg_pooling(data, mode='2d', name=None):
    with tf.name_scope(name):
        if mode == '2d' : 
            global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        elif mode == '3d' : 
            global_avg_pool = tf.keras.layers.GlobalAveragePooling3D()
        else :
            raise ValueError("'mode' must be '2d' or '3d'.")
        return global_avg_pool(data)
    
def dropout(data, ratio, name=None):
    with tf.name_scope(name):
        return tf.nn.dropout(data, ratio, name=name)

def lrn(data, depth_radius, alpha, beta, name=None):
    with tf.name_scope(name):
        return tf.nn.local_response_normalization(data, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=1.0, name=name)

def batch_norm(data, is_train, name=None, data_format='channels_last', 
               USE_FUSED_BN = True, BN_EPSILON = 0.001, BN_MOMENTUM = 0.99):
    
    bn_axis = -1 if data_format == 'channels_last' else 1
    
    with tf.name_scope(name):
        return tf.layers.batch_normalization(data, training=is_train, name=name, momentum=BN_MOMENTUM, axis=bn_axis,
                                             epsilon=BN_EPSILON, reuse=None, fused=USE_FUSED_BN)

def fc(data, num_out, name=None, relu=True, bn=True, use_bias=False, is_train=None):
    if bn and is_train==None:
        raise ValueError('Phase should be declared "True" for train or "False" for test when you activate batch normalization.')
        
    with tf.variable_scope(name) as scope:
        output = tf.layers.dense(inputs=data, use_bias=use_bias, units=num_out)
        if bn : 
            output = batch_norm(output, is_train)
        if relu : 
            output = tf.nn.relu(output)
    return output

def global_map_average_pooling(data, mode='2d', name=None):
    with tf.variable_scope(name):
        if mode == '2d' :
            _, x, y, _ = data.get_shape().as_list()
            flatten_len = x*y 
        if mode == '3d' :
            _, x, y, z, _ = data.get_shape().as_list()
            flatten_len = x*y*z

        data = tf.reshape(data, (-1, flatten_len, c))
        data = [tf.reduce_mean(data[:,idx,:], axis=-1) for idx in range(flatten_len)]
        data = [tf.expand_dims(fmap, axis=-1) for fmap in data]
        data = tf.concat(data, axis=-1)
        
        dimension = (-1,x,y,1) if mode == '2d' else (-1,x,y,z,1)
        data = tf.reshape(data, dimension)

        return data

def squeeze_excitation_layer(data, ratio, mode='2d', name=None):
    with tf.variable_scope(name):
        c = data.get_shape().as_list()[-1]
        squeeze = global_avg_pooling(data, mode)

        excitation = fc(squeeze, c/ratio, bn=False, relu=True, name='fc1')
        excitation = fc(excitation, c, bn=False, relu=False, name='fc2')
        excitation = tf.nn.sigmoid(excitation)

        dimension = (-1,1,1,c) if mode == '2d' else (-1,1,1,1,c)
        excitation = tf.reshape(excitation, dimension)

        scale = data * excitation

        return scale
    
def positional_squeeze_excitation_layer(data, ratio, mode='2d', name=None):
    with tf.variable_scope(name):
        if mode == '2d' :
            _, x, y, _ = data.get_shape().as_list()
        if mode == '3d' :
            _, x, y, z, _ = data.get_shape().as_list()
            
        squeeze = global_map_average_pooling(data, mode)
        flatten_len = x*y if mode == '2d' else x*y*z
        squeeze = tf.reshape(squeeze, (-1, flatten_len))
        
        excitation = fc(squeeze, flatten_len/ratio, bn=False, relu=True, name='fc1')
        excitation = fc(excitation, flatten_len, bn=False, relu=False, name='fc2')
        excitation = tf.nn.sigmoid(excitation)
        
        dimension = (-1,x,y,1) if mode == '2d' else (-1,x,y,z,1)
        excitation = tf.reshape(excitation, dimension)
        
        scale = data * excitation
        
        return scale