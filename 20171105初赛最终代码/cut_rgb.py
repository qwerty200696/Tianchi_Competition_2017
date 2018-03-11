from collections import defaultdict
import csv
import sys
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib import cm

# FILE_2015 = './preliminary/quickbird2015.tif'
# FILE_2017 = './preliminary/quickbird2017.tif'
# FILE_cadastral2015 = './20170907_hint/cadastral2015.tif'
# FILE_tinysample = './20170907_hint/tinysample.tif'
FILE_2015 = 'E:/Tianchi/NEW_DATA/second/quickbird2015_preliminary_2.tif'
FILE_2017 = 'E:/Tianchi/NEW_DATA/second/quickbird2017_preliminary_2.tif'
# FILE_cadastral2015 = './20170907_hint/cadastral2015.tif'
# FILE_tinysample = './20170907_hint/tinysample.tif'




im_2015 = tiff.imread(FILE_2015)#.transpose([1, 2, 0])
im_2017 = tiff.imread(FILE_2017)#.transpose([1, 2, 0])
# im_tiny = tiff.imread(FILE_tinysample)
# im_cada = tiff.imread(FILE_cadastral2015)




def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    # print(matrix.shape)
    return matrix

# img_size = 224 # 15106/ 256 =59...2  5106/256=19..284
img_size = 224 # 15106/ 256 =59...2  8106/256=19..284
a=int(len(im_2015)/img_size+1)
b=int(len(im_2015[0])/img_size)
rgb2015 = np.zeros((224,224,3))
rgb2017 = np.zeros((224,224,3))
for i in range(int(len(im_2015)/img_size)+1): # last 284
    for j in range(int(len(im_2015[0])/img_size)+1): #last 2 too small, drop one
        im_name = str(i)+'_'+str(j)+'_'+str(img_size)+'_.png'
        r2015=scale_percentile(im_2015[i * img_size:i * img_size + img_size, j * img_size:j * img_size + img_size, 0:1]) * 255
        g2015=scale_percentile(im_2015[i * img_size:i * img_size + img_size, j * img_size:j * img_size + img_size, 1:2]) * 255
        b2015 = scale_percentile(im_2015[i * img_size:i * img_size + img_size, j * img_size:j * img_size + img_size, 2:3]) * 255
        r2017 = scale_percentile(im_2017[i * img_size:i * img_size + img_size, j * img_size:j * img_size + img_size, 0:1]) * 255
        g2017 = scale_percentile(im_2017[i * img_size:i * img_size + img_size, j * img_size:j * img_size + img_size, 1:2]) * 255
        b2017 = scale_percentile(im_2017[i * img_size:i * img_size + img_size, j * img_size:j * img_size + img_size, 2:3]) * 255
        # cv2.imwrite("E:/Tianchi/NEW_DATA/224_rgb/2015_224_rgb/"+im_name,scale_percentile(im_2015[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size, 0:1])*255)
        # cv2.imwrite("E:/Tianchi/NEW_DATA/224_rgb/2017_224_rgb/"+im_name,scale_percentile(im_2017[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size, 0:3])*255)
        # cv2.imwrite("unet_test_pic/cada_960/"+im_name,im_cada[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size]*255)
        # cv2.imwrite("unet_test_pic/tiny_960/"+im_name,im_tiny[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size]*255)
        rgb2015=np.stack([r2015,g2015,b2015],axis = 2)
        rgb2015 = np.squeeze(rgb2015)
        # rgb2017 = [r2017,g2017,b2017]
        cv2.imwrite("E:/Tianchi/NEW_DATA/224_rgb/2015_224_r/"+im_name,rgb2015)
        # cv2.imwrite("E:/Tianchi/NEW_DATA/224_rgb/2017_224_rgb/"+im_name,scale_percentile(im_2017[i*img_size:i*img_size+img_size, j*img_size:j*img_size+img_size, 0:3])*25