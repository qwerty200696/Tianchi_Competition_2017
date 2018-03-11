# coding: utf-8
'''
    - train "ZF_UNET_224" CNN with random images
'''

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import cv2
import random
import numpy as np
import pandas as pd
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import __version__
from zf_unet_224_model import *


def batch_generator(batch_size):
    input_path='E:/Tianchi/train_data/orig/'
    label_path='E:/Tianchi/train_data/bz/'
    image_number = len(os.listdir(label_path))
    while True:
        image_number_shuffle= np.arange(image_number)
        np.random.shuffle(image_number_shuffle)
        for img_name_beg in range(0,image_number-batch_size,batch_size):
            image_list = []
            mask_list = []
            for img_name_index in range(batch_size):
                img_path_input = input_path + os.listdir(label_path)[image_number_shuffle[img_name_beg+img_name_index]]
                img_path_label = label_path + os.listdir(label_path)[image_number_shuffle[img_name_beg+img_name_index]]
                img_input = cv2.imread(img_path_input)
                img_label = cv2.imread(img_path_label)
                height,width,shape=img_input.shape
                if width==223:
                    print('name:',img_path_input)
                img_label = img_label[:,:,0]
                img_label[img_label>100]=255
                img_label[img_label<=100]=0
                image_list.append(img_input)
                mask_list.append([img_label])

            image_list = np.array(image_list, dtype=np.float32)
            if K.image_dim_ordering() == 'th':
                image_list = image_list.transpose((0, 3, 1, 2))
            image_list = preprocess_batch(image_list)
            mask_list = np.array(mask_list, dtype=np.float32)
            mask_list /= 255.0
            yield image_list, mask_list


def batch_generator_test(batch_size):
    input_path='E:/Tianchi/train_data/orig_verify/'
    label_path='E:/Tianchi/train_data/bz_verify/'
    image_number = len(os.listdir(label_path))
    while True:
        image_number_shuffle= np.arange(image_number)
        np.random.shuffle(image_number_shuffle)
        for img_name_beg in range(0,image_number-batch_size,batch_size):
            image_list = []
            mask_list = []
            for img_name_index in range(batch_size):
                img_path_input = input_path + os.listdir(label_path)[image_number_shuffle[img_name_beg+img_name_index]]
                img_path_label = label_path + os.listdir(label_path)[image_number_shuffle[img_name_beg+img_name_index]]
                img_input = cv2.imread(img_path_input)
                img_label = cv2.imread(img_path_label)
                height,width,shape=img_input.shape
                if width==223:
                    print('name:',img_path_input)
                img_label = img_label[:,:,0]
                img_label[img_label>100]=255
                img_label[img_label<=100]=0
                image_list.append(img_input)
                mask_list.append([img_label])

            image_list = np.array(image_list, dtype=np.float32)
            if K.image_dim_ordering() == 'th':
                image_list = image_list.transpose((0, 3, 1, 2))
            image_list = preprocess_batch(image_list)
            mask_list = np.array(mask_list, dtype=np.float32)
            mask_list /= 255.0
            yield image_list, mask_list



def train_unet():
    output_model_path='zf_unet_224_temp_vv4_0.791.h5'
    epochs = 150
    patience = 1000
    batch_size = 16
    optim_type = 'Adam'
    learning_rate = 0.0001
    model = ZF_UNET_224()
    if os.path.isfile(output_model_path):
        model.load_weights(output_model_path)

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('zf_unet_224_temp_vv5_best.h5', monitor='val_loss', save_best_only=True, verbose=0),
        ModelCheckpoint('zf_unet_224_temp_vv5_false.h5', monitor='val_loss', save_best_only=False, verbose=0),
    ]

    print('Start training...')
    history = model.fit_generator(
        generator=batch_generator(batch_size),
        epochs=epochs,
        steps_per_epoch=737,
        validation_data=batch_generator(batch_size),
        validation_steps=100,
        verbose=2,
        callbacks=callbacks)

    model.save_weights(output_model_path)
    pd.DataFrame(history.history).to_csv('zf_unet_224_train.csv', index=False)
    print('Training is finished (weights zf_unet_224.h5 and log zf_unet_224_train.csv are generated )...')


if __name__ == '__main__':
    if K.backend() == 'tensorflow':
        try:
            from tensorflow import __version__ as __tensorflow_version__
            print('Tensorflow version: {}'.format(__tensorflow_version__))
        except:
            print('Tensorflow is unavailable...')
    else:
        try:
            from theano.version import version as __theano_version__
            print('Theano version: {}'.format(__theano_version__))
        except:
            print('Theano is unavailable...')
    print('Keras version {}'.format(__version__))
    print('Dim ordering:', K.image_dim_ordering())
    train_unet()
