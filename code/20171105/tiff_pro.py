import cv2
import numpy as np
import tifffile as tiff
import os

# path='E:/Tianchi/unet_test_pic/unet_test_all_minpic_vv4_red_all/'
path = 'E:/Tianchi/NEW_DATA2/224_rgb/pinjie_new/'
store_path = 'E:/Tianchi/NEW_DATA2/224_rgb/pinjie_new.tif'
height = 3000
width = 15106
img_all = os.listdir(path)
img_number = len(img_all)
img_cut = 224
tiff_final = np.zeros((height, width), dtype=np.uint8)
for i in range(img_number):

    img_height = int(img_all[i].split('_')[0])
    img_width = int(img_all[i].split('_')[1].split('.')[0])
    # print('height',img_height)
    # print('width',img_width)
    img_single = cv2.imread(path + img_all[i], 0)
    img_single[img_single <= 30] = 0
    img_single[img_single > 30] = 255
    # height,width=img_single.shape

    height_max = int(height / img_cut)
    width_max = int(width / img_cut)
    if img_height == height_max and img_width != width_max:
        # error！！！！！
        # print('height',img_height,'width',img_width,img_all[i])
        img_single = cv2.resize(img_single, (img_cut, height - img_cut * height_max), interpolation=cv2.INTER_CUBIC)
        tiff_final[img_height * img_cut:img_height * img_cut + img_cut,
        img_width * img_cut:img_width * img_cut + img_cut] = img_single
        # tiff_final[img_height * img_cut:img_height * img_cut + img_cut,img_width * img_cut:img_width * img_cut + img_cut] = np.zeros((178,width))
    if img_width == width_max and img_height != height_max:
        img_single = cv2.resize(img_single, (width - width_max * img_cut, img_cut), interpolation=cv2.INTER_CUBIC)
        tiff_final[img_height * img_cut:img_height * img_cut + img_cut,
        img_width * img_cut:img_width * img_cut + img_cut] = img_single
    if img_width == width_max and img_height == height_max:
        img_single = cv2.resize(img_single, (width - width_max * img_cut, height - img_cut * height_max),
                                interpolation=cv2.INTER_CUBIC)
        tiff_final[img_height * img_cut:img_height * img_cut + img_cut,
        img_width * img_cut:img_width * img_cut + img_cut] = img_single
    if img_width != width_max and img_height != height_max:
        # img_single = cv2.resize(img_single, (98, 178), interpolation=cv2.INTER_CUBIC)
        tiff_final[img_height * img_cut:img_height * img_cut + img_cut,
        img_width * img_cut:img_width * img_cut + img_cut] = img_single
        # if img_height==22:
        #     # img_single = cv2.resize(img_single, (224, 178), interpolation=cv2.INTER_CUBIC)
        #     tiff_final[img_height * img_cut:img_height * img_cut + img_cut,img_width * img_cut:img_width * img_cut + img_cut] = img_single[:178,:]
        # else:
        #     tiff_final[img_height * img_cut:img_height * img_cut + img_cut,img_width * img_cut:img_width * img_cut + img_cut] = img_single
tiff.imsave(store_path, tiff_final)
