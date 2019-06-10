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

def max_pooling(data, ksize=3, ssize=2, name=None):
    with tf.name_scope(name):
        return tf.nn.max_pool(data, ksize=[1,ksize,ksize,1], strides=[1,ssize,ssize,1], padding="SAME", name=name)

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

def batch_norm(data, is_train, name=None):
    with tf.name_scope(name):
        return tf.layers.batch_normalization(data, training=is_train, name=name)

def fc(data, num_out, name=None, relu=True, bn=True, is_train=None):
    if bn and is_train==None:
        raise ValueError('Phase should be declared "True" for train or "False" for test when you activate batch normalization.')
        
    with tf.variable_scope(name) as scope:
        output = tf.layers.dense(inputs=data, use_bias=True, units=num_out)
        if bn : 
            output = batch_norm(output, is_train)
        if relu : 
            output = tf.nn.relu(output)
    return output
