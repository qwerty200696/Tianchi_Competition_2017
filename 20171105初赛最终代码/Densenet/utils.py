import tensorflow as tf
import math
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

######是否要加偏置
'''
def multi_dilate(x):
    conv1=tf.nn.relu(conv2d(x,3,64,3,1))
    conv2_rate1=conv2d(conv1,64,64,3,1)
    conv2_rate2=tf.nn.atrous_conv2d(conv1,filters=(3,3,64,64),rate=2,padding='SAME')
    conv2_rate4 = tf.nn.atrous_conv2d(conv1, filters=(3, 3, 64, 64), rate=4, padding='SAME')
    conv2_rate6 = tf.nn.atrous_conv2d(conv1, filters=(3, 3, 64, 64), rate=6, padding='SAME')
    conv2_concat = tf.concat([conv2_rate1,conv2_rate2,conv2_rate4,conv2_rate6],axis=2)
    conv2_BN = batch_norm_layer(conv2_concat)
    return tf.nn.relu(conv2_BN)

def residual(x, filters, kernel, strides,is_training=True):
    with tf.variable_scope('residual') as scope:
        conv1 = conv2d(x, filters, filters, kernel, strides, name='conv1')
        conv1_BN = batch_norm_layer(conv1,is_training,'res_conv1BN')

        conv2 = conv2d(tf.nn.relu(conv1_BN), filters, filters, kernel, strides, name='conv2')
        conv2_BN = batch_norm_layer(conv2,is_training,'res_conv2BN')

        residual =tf.nn.relu(x + conv2_BN)
        return residual

'''


def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT', name='conv'):
    with tf.variable_scope(name) as scope:
        shape = [kernel, kernel, input_filters, output_filters]
        # weight = tf.get_variable('weight', initializer=tf.random_uniform(shape, minval=-math.sqrt(6) / (
        # input_filters + output_filters), maxval=math.sqrt(6) / (input_filters + output_filters)))
        weight = tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.constant(0.0, shape=[output_filters])
        bias = tf.get_variable('bias', initializer=bias)
        x_conv = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME', name='conv')
        return tf.nn.bias_add(x_conv,bias)

# def batch_norm_layer(x,train_phase,scope_bn):
#     bn_train = batch_norm(x,decay=0.99,center=True,scale=True,updates_collections=None,is_training=train_phase,reuse=None,trainable=True,scope=scope_bn+'train')
#     return bn_train

def batch_norm_layer(x, train_phase):
    bn_train = batch_norm(x, decay=0.99, center=True, scale=True, updates_collections=None, is_training=train_phase,
                          reuse=None, trainable=True, scope='bn')
    return bn_train


def bn_relu_conv(image, is_training, input_filters, output_filters, kernel, strides,name='bn_relu_conv1'):
    with tf.variable_scope(name) as scope:
        image_bn = batch_norm_layer(image, train_phase=is_training)
        image_relu = tf.nn.relu(image_bn)
        image_conv = conv2d(image_relu, input_filters, output_filters, kernel, strides)
    return image_conv


def add_layer(name, l, is_training, input_filters1, input_filters2=64, output_filters1=64, output_filters2=16,
              kernel1=1, kernel2=3, strides=1):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        c = bn_relu_conv(l, is_training, input_filters1, output_filters1, kernel1, strides,name='bn_relu_conv1')
        c = bn_relu_conv(c, is_training, input_filters2, output_filters2, kernel2, strides,name='bn_relu_conv2')
        l = tf.concat([c, l], 3)
    return l


def add_transition_average(name, l, is_training, input_filters, output_filters):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        l = bn_relu_conv(l, is_training, input_filters, output_filters, 1, 1)
        l_pool = tf.nn.avg_pool(l,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
    return l, l_pool


def upsample(l, input_filters, output_filters, kernel, strides):
    shape = l.get_shape().as_list()
    shape[1] = shape[1] * 2
    shape[2] = shape[2] * 2
    shape[3] = output_filters
    weight_shape = [kernel, kernel, input_filters, output_filters]
    # weight = tf.get_variable('weight', initializer=tf.random_uniform(weight_shape, minval=-math.sqrt(6) / (
    # input_filters + output_filters), maxval=math.sqrt(6) / (input_filters + output_filters)))
    weight = tf.get_variable('weight', shape=weight_shape, initializer=tf.contrib.layers.xavier_initializer())
    upsample_result = tf.nn.conv2d_transpose(l, weight,
                                             output_shape=shape,
                                             strides=[1, strides, strides, 1], padding="SAME")
    return upsample_result
