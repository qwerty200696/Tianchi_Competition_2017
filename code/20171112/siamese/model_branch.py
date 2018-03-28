
#两张图片分别输入densenet-unet中，将输出concat
import tensorflow as tf
import math
from utils import *
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

dense_block1_num=2
dense_block2_num=2
dense_block3_num=3
dense_block4_num=3
growth_rate=16

def dense_net(image,is_training=True):
    with tf.variable_scope('conv1') as scope:
        l = conv2d(image,3,24,3,1)

    with tf.variable_scope('conv2_3') as scope:
        l_big = bn_relu_conv(l, is_training, 24, 32, 3, 1, name='bn_relu_conv1')
        # l = bn_relu_conv(l, is_training, 32, 32, 3, 1, name='bn_relu_conv2')


    l_first_down = tf.nn.max_pool(l_big,[1,2,2,1], strides=[1,2,2,1],padding='SAME')
    with tf.variable_scope('block1') as scope:
        # l = conv2d(l_first_down,32,growth_rate,3,1)#delete
        l = l_first_down
        for i in range(dense_block1_num):
            l = add_layer('dense_layer.{}'.format(i), l,is_training,input_filters1=growth_rate*i+32)
        # l = bn_relu_conv(l, is_training, growth_rate*(dense_block1_num+1), 32, 3, 1)
        block1,l = add_transition_average('transition1', l,is_training,input_filters=growth_rate*dense_block1_num+32,output_filters=32)

    with tf.variable_scope('block2') as scope:
        # l = conv2d(l,32,growth_rate,3,1)#delete
        for i in range(dense_block2_num):
            l = add_layer('dense_layer.{}'.format(i), l,is_training,input_filters1=growth_rate*i+32)
        # l = bn_relu_conv(l,is_training,growth_rate*(1+dense_block2_num),32,3,1)
        block2,l = add_transition_average('transition2', l,is_training,input_filters=growth_rate*dense_block2_num+32,output_filters=32)

    with tf.variable_scope('block3') as scope:
        # l = conv2d(l, 32, growth_rate, 3, 1)
        for i in range(dense_block3_num):
            l = add_layer('dense_layer.{}'.format(i), l, is_training, input_filters1=growth_rate * i + 32)
        block3,l = add_transition_average('transition3', l,is_training,input_filters=growth_rate*dense_block3_num+32,output_filters=32)

    with tf.variable_scope('block4') as scope:
        # l = conv2d(l, 32, growth_rate, 3, 1)
        for i in range(dense_block4_num):
            l = add_layer('dense_layer.{}'.format(i), l, is_training, input_filters1=growth_rate * i + 32)


    with tf.variable_scope('block3_up') as scope:
        l = bn_relu_conv(l, is_training, growth_rate * dense_block4_num+32, 32, 3, 1,name='bn_relu_conv1')
        l=upsample(l,32,32,3,2)
        l=tf.concat([l,block3],3)
        # l=bn_relu_conv(l,is_training,64,growth_rate,3,1,name='bn_relu_conv2')
        for i in range(dense_block3_num):
            l=add_layer('dense_layer.{}'.format(i),l,is_training,input_filters1=growth_rate*i+64)

    with tf.variable_scope('block2_up') as scope:
        l = bn_relu_conv(l, is_training, growth_rate *dense_block3_num+64, 32, 3, 1,name='bn_relu_conv1')
        l=upsample(l,32,32,3,2)
        l=tf.concat([l,block2],3)
        # l = bn_relu_conv(l, is_training, 64, growth_rate, 3, 1,name='bn_relu_conv2')
        for i in range(dense_block2_num):
            l=add_layer('dense_layer.{}'.format(i),l,is_training,input_filters1=growth_rate*i+64)


    with tf.variable_scope('block1_up') as scope:
        l = bn_relu_conv(l, is_training, growth_rate *dense_block2_num+64, 32, 3, 1,name='bn_relu_conv1')
        l=upsample(l,32,32,3,2)
        l=tf.concat([l,block1],3)
        # l = bn_relu_conv(l, is_training, 64, growth_rate, 3, 1,name='bn_relu_conv2')
        for i in range(dense_block1_num):
            l=add_layer('dense_layer.{}'.format(i),l,is_training,input_filters1=growth_rate*i+64)


    l = bn_relu_conv(l, is_training, growth_rate * dense_block1_num+64, 32, 3, 1,name='bn_relu_conv1')
    with tf.variable_scope('upsample1') as scope:
        l=upsample(l,32,32,3,2)
        #concat
        l=tf.concat([l,l_big],3)
        l=bn_relu_conv(l,is_training,64,64,3,1)
        l = tf.nn.dropout(l,0.5)
        #spatial dropout,dropout rate 0.5
    l = bn_relu_conv(l,is_training,64,32,1,1,name='bn_relu_conv2')
    return l





def inference(image15,image17,is_training=True,scope_name='inference_net'):
    with tf.variable_scope(scope_name) as scope:
        annotation_pred15 = dense_net(image15,is_training)
        scope.reuse_variables()
        annotation_pred17 = dense_net(image17, is_training)

        pred_all = tf.concat([annotation_pred15,annotation_pred17],3)
    with tf.variable_scope('final') as scope:
        pred_conv1 = bn_relu_conv(pred_all,is_training,64,32,1,1,name='conv1')
        pred_conv2 = bn_relu_conv(pred_conv1,is_training,32,1,1,1,name='conv2')

        pred = tf.nn.sigmoid(pred_conv2)


    return pred