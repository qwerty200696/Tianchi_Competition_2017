import model
import tensorflow as tf
import time
import os
import numpy as np
import argparse
import cv2
from utils import *
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm






dense_block1_num=2
dense_block2_num=2
dense_block3_num=3
dense_block4_num=3
growth_rate=16

def dense_net(image,is_training=True):
    with tf.variable_scope('conv1') as scope:
        l = conv2d(image,6,48,3,1)

    with tf.variable_scope('conv2_3') as scope:
        l_big = bn_relu_conv(l, is_training, 48, 32, 3, 1, name='bn_relu_conv1')
        l_big2 = bn_relu_conv(l_big, is_training, 32, 32, 3, 2, name='bn_relu_conv2')
        l = bn_relu_conv(l_big2, is_training, 32, 32, 3, 1, name='bn_relu_conv3')


    l_first_down = tf.nn.max_pool(l,[1,2,2,1], strides=[1,2,2,1],padding='SAME')
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

    with tf.variable_scope('block2_up') as scope:
        l = bn_relu_conv(l, is_training, growth_rate * dense_block3_num+32, 32, 3, 1,name='bn_relu_conv1')
        l=upsample(l,32,32,3,2)
        l=tf.concat([l,block2],3)
        # l=bn_relu_conv(l,is_training,64,growth_rate,3,1,name='bn_relu_conv2')
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
        l=tf.concat([l,l_big2],3)
        l=bn_relu_conv(l,is_training,64,32,3,1,name='bn_relu_conv1')
    with tf.variable_scope('upsample2') as scope:
        l=upsample(l,32,32,3,2)
        l=tf.concat([l,l_big],3)
        l = bn_relu_conv(l, is_training, 64, 64, 3, 1,name='bn_relu_conv1')
        l = tf.nn.dropout(l,0.5)
        #spatial dropout,dropout rate 0.5
    l = bn_relu_conv(l,is_training,64,32,1,1,name='bn_relu_conv2')
    return l





def inference(image15,image17,is_training=True,scope_name='inference_net',scope_reuse=False):
    with tf.variable_scope(scope_name, reuse=scope_reuse) as scope:
        if scope_reuse:
            scope.reuse_variables()
        image_all = tf.concat([image15,image17],3)
        annotation_pred = dense_net(image_all,is_training)
        with tf.variable_scope('final') as scope:
            pred_conv1 = bn_relu_conv(annotation_pred,is_training,32,1,1,1,name='conv1')
            # pred = tf.nn.sigmoid(pred_conv1)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            if img_name_index==0:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('model restored')
        return pred_conv1











sess = tf.InteractiveSession()
height = 224
width = 224
batch_size = 10
write_number = 0
gamma = 2
alpha = 2
is_training = True
previous_time = time.clock()
total_loss_list = []
create_record()
records_path = os.path.join(FLAGS.buckets, "train.tfrecords")
img15,img17, label = read_and_decode(records_path)





#模型位置
ckpt=tf.train.get_checkpoint_state('E:/Tianchi/Densenet/my_network/model/')

#15,17位置
path15='E:/Tianchi/final_data/224_rgb/2017_224_rgb/'
path17 =''  # 1106-changed
#结果位置
write_path='E:/Tianchi/Densenet/my_network/result_newmodel45000_2017_224/'

image_number = len(os.listdir(path15))
write_number=write_number+1
output_store = np.zeros([batch_size,224,224,3])
for img_name_index in range(0,image_number,batch_size):
    if img_name_index+batch_size>image_number:
        batch_size = image_number-img_name_index
    img_batch_store15 = np.zeros([batch_size, 224, 224, 3])
    img_batch_store17 = np.zeros([batch_size, 224, 224, 3])
    for i in range(batch_size):
        img_path_input15 = path15 + os.listdir(path15)[img_name_index+i]
        img_test15 = cv2.imread(img_path_input15)
        img_path_input17 = path17 + os.listdir(path)[img_name_index+i]
        img_test17 = cv2.imread(img_path_input17)


        # height,width,channel = img_test.shape
        img_test15 = cv2.resize(img_test15,(224,224),interpolation=cv2.INTER_CUBIC)
        img_batch_store15[i,:,:,:]= img_test15
        img_test17 = cv2.resize(img_test17,(224,224),interpolation=cv2.INTER_CUBIC)
        img_batch_store17[i,:,:,:]= img_test17


    img_test_tensor15 = tf.convert_to_tensor(img_batch_store15, dtype=tf.uint8)
    img_input15 = tf.reshape(img_test_tensor15, [batch_size, 224, 224, 3])
    img_input15 = tf.cast(img_input15, tf.float32)
    img_input15 = img_input15 * (1. / 255)
    img_test_tensor17 = tf.convert_to_tensor(img_batch_store17, dtype=tf.uint8)
    img_input17 = tf.reshape(img_test_tensor17 [batch_size, 224, 224, 3])
    img_input17 = tf.cast(img_input17, tf.float32)
    img_input17 = img_input17 * (1. / 255)

    if img_name_index==0:
        output = inference(img_input15，img_input17,img_name_index,is_training=False,scope_reuse=False)
    else:
        output = inference(img_input15,img_input17,img_name_index,is_training=False,scope_reuse=True)
        # output = inference(img_input, is_training=True,scope_reuse=True)
        # output =model_enhance_subpixel_BN.transform_net(img_input,size, upscale,scope_reuse=True,is_training=False)
    output = output * 255
    output = tf.reshape(output, [batch_size,224,224, 1])
    output = output.eval()
    output[output > 100] = 255
    output[output <= 100] = 0

    for j in range(batch_size):
        savepath = write_path + os.listdir(path)[img_name_index+j]
        # output1 = cv2.resize(output[j,:,:,:], (224,224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(savepath,output[j,:,:,:])





