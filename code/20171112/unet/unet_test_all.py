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
from keras.preprocessing import image
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import __version__
from zf_unet_224_model import *


def gen_random_image():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=np.uint8)

    # Background
    dark_color0 = random.randint(0, 100)
    dark_color1 = random.randint(0, 100)
    dark_color2 = random.randint(0, 100)
    img[:, :, 0] = dark_color0
    img[:, :, 1] = dark_color1
    img[:, :, 2] = dark_color2

    # Object
    light_color0 = random.randint(dark_color0 + 1, 255)
    light_color1 = random.randint(dark_color1 + 1, 255)
    light_color2 = random.randint(dark_color2 + 1, 255)
    center_0 = random.randint(0, 224)
    center_1 = random.randint(0, 224)
    r1 = random.randint(10, 56)
    r2 = random.randint(10, 56)
    cv2.ellipse(img, (center_0, center_1), (r1, r2), 0, 0, 360, (light_color0, light_color1, light_color2), -1)
    cv2.ellipse(mask, (center_0, center_1), (r1, r2), 0, 0, 360, 255, -1)

    # White noise
    density = random.uniform(0, 0.1)
    for i in range(224):
        for j in range(224):
            if random.random() < density:
                img[i, j, 0] = random.randint(0, 255)
                img[i, j, 1] = random.randint(0, 255)
                img[i, j, 2] = random.randint(0, 255)

    return img, mask


def batch_generator(batch_size):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            img, mask = gen_random_image()
            image_list.append(img)
            mask_list.append([mask])

        image_list = np.array(image_list, dtype=np.float32)
        if K.image_dim_ordering() == 'th':
            image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_batch(image_list)
        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list /= 255.0
        yield image_list, mask_list


def train_unet():
    out_model_path = 'zf_unet_224_temp_vv4_0.791.h5'
    batch_size = 12
    model = ZF_UNET_224()
    if os.path.isfile(out_model_path):
        model.load_weights(out_model_path)
    print('model stored!')
    pic_dir = 'E:/Tianchi/satellite_img/2015_224_red/'
    paths = os.listdir(pic_dir)
    for path1 in paths:
        # img_path='E:/Tianchi/unet_test_pic/2015/0_1_224_.jpg'
        # img=image.load_img(pic_dir+path1,target_size=(224,224))
        # img = image.load_img(pic_dir + path1)
        # x=image.img_to_array(img)
        # x90 = np.rot90(x,1)
        # x180=np.rot90(x,2)
        x = cv2.imread(pic_dir + path1)
        x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_CUBIC)
        m90 = cv2.getRotationMatrix2D((112, 112), 90, 1)
        m180 = cv2.getRotationMatrix2D((112, 112), 180, 1)
        m270 = cv2.getRotationMatrix2D((112, 112), 270, 1)
        x90 = cv2.warpAffine(x, m90, (224, 224))
        x180 = cv2.warpAffine(x, m180, (224, 224))
        x270 = cv2.warpAffine(x, m270, (224, 224))

        x = np.expand_dims(x, axis=0)
        x = np.array(x, dtype=np.float32)
        x = x / 255
        x = x - 0.5
        preds = model.predict(x)

        x90 = np.expand_dims(x90, axis=0)
        x90 = np.array(x90, dtype=np.float32)
        x90 = x90 / 255
        x90 = x90 - 0.5
        preds90 = model.predict(x90)
        preds90 = np.squeeze(preds90)
        preds90 = cv2.warpAffine(preds90, m270, (224, 224))

        x180 = np.expand_dims(x180, axis=0)
        x180 = np.array(x180, dtype=np.float32)
        x180 = x180 / 255
        x180 = x180 - 0.5
        preds180 = model.predict(x180)
        preds180 = np.squeeze(preds180)
        preds180 = cv2.warpAffine(preds180, m180, (224, 224))

        x270 = np.expand_dims(x270, axis=0)
        x270 = np.array(x270, dtype=np.float32)
        x270 = x270 / 255
        x270 = x270 - 0.5
        preds270 = model.predict(x270)
        preds270 = np.squeeze(preds270)
        preds270 = cv2.warpAffine(preds270, m90, (224, 224))

        preds = (preds + preds90 + preds180 + preds270) / 4

        preds = preds * 255
        preds[preds > 150] = 255
        preds[preds <= 150] = 0
        preds = preds.astype(np.uint8)
        cv2.imwrite('E:/Tianchi/satellite_img/v6_false/2015_224_red_result/' + path1, preds)


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
