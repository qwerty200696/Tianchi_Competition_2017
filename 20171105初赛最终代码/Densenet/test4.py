import tensorflow as tf
import cv2
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import *

dense_block1_num=2
dense_block2_num=2
dense_block3_num=3
dense_block4_num=3
growth_rate=16
test_number=0
def dense_net(image,img_name_index, is_training=True):
    with tf.variable_scope('conv1') as scope:
        l = conv2d(image,3,24,3,1)

    with tf.variable_scope('conv2_3') as scope:
        l_big = bn_relu_conv(l, is_training, 24, 32, 3, 1, name='bn_relu_conv1')
        # l = bn_relu_conv(l, is_training, 32, 32, 3, 1, name='bn_relu_conv2')


#跳链接层数可以多，其他地方尽量少
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


    with tf.variable_scope('bn_sigmoid_conv') as scope:
        l=bn_relu_conv(l,is_training,32,1,1,1)

        image_conv = tf.nn.sigmoid(l)


    saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        if img_name_index==0:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('model restored')
    return image_conv


def inference(image,img_name_index,is_training=True, scope_name='inference_net', scope_reuse=True):
    with tf.variable_scope(scope_name, reuse=scope_reuse) as scope:
        if scope_reuse:
            scope.reuse_variables()
        annotation_pred = dense_net(image, img_name_index,is_training)
        return annotation_pred



sess = tf.InteractiveSession()
# height = 960#训练图片的高
# width = 960#训练图片的宽
batch_size = 40
write_number=0
is_training=True
previous_time = time.clock()
total_loss_list=[]
ckpt=tf.train.get_checkpoint_state('E:/Tianchi/Densenet/my_network/model/')

path='E:/Tianchi/NEW_DATA2/224_rgb/2015_224_rgb/'
write_path='E:/Tianchi/Densenet/my_network/result_newmodel45000_2015_224/'
image_number = len(os.listdir(path))
write_number=write_number+1
output_store = np.zeros([batch_size,224,224,3])
for img_name_index in range(0,image_number,batch_size):
    if img_name_index+batch_size>image_number:
        batch_size = image_number-img_name_index
    img_batch_store = np.zeros([batch_size, 224, 224, 3])
    for i in range(batch_size):
        img_path_input = path + os.listdir(path)[img_name_index+i]
        img_test = cv2.imread(img_path_input)
        # height,width,channel = img_test.shape
        img_test = cv2.resize(img_test,(224,224),interpolation=cv2.INTER_CUBIC)
        img_batch_store[i,:,:,:]= img_test

    img_test_tensor = tf.convert_to_tensor(img_batch_store, dtype=tf.uint8)
    img_input = tf.reshape(img_test_tensor, [batch_size, 224, 224, 3])
    img_input = tf.cast(img_input, tf.float32)
    img_input = img_input * (1. / 255)
    if img_name_index==0:
        output = inference(img_input,img_name_index,is_training=False,scope_reuse=False)
    else:
        output = inference(img_input,img_name_index,is_training=False,scope_reuse=True)
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

