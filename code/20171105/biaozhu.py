import cv2
import os

DATA_DIR = "./data"

IM_ROWS = 3000
IM_COLS = 15106
ROI_SIZE = 224
import numpy as np


def on_mouse(event, x, y, flags, params):
    img, points = params['img'], params['points']
    if event == cv2.EVENT_FLAG_LBUTTON:
        points.append((x, y))

    if event == cv2.EVENT_FLAG_RBUTTON:
        points.pop()

    temp = img.copy()
    if len(points) > 2:
        cv2.fillPoly(temp, [np.array(points)], (0, 0, 255))

    for i in range(len(points)):
        cv2.circle(temp, points[i], 1, (0, 0, 255))

    cv2.circle(temp, (x, y), 1, (0, 255, 0))
    cv2.namedWindow("img", 0)
    cv2.resizeWindow('img', 1024, 1024)
    cv2.imshow('img', temp)


def label_img(img_1, img_2, label_name_1, label_name_2):
    c = 'x'
    tiny_2015 = np.zeros(img_1.shape)
    tiny_2017 = np.zeros(img_2.shape)
    while c != 'n':
        points = []

        cv2.namedWindow('img_2015', 0)
        temp_1 = img_1.copy()
        cv2.setMouseCallback('img_2015', on_mouse, {'img': temp_1, 'points': points})
        cv2.imshow('img_2015', img_1)

        cv2.namedWindow('img_2017', 0)
        temp_2 = img_2.copy()
        cv2.setMouseCallback('img_2017', on_mouse, {'img': temp_2, 'points': points})
        cv2.imshow('img_2017', img_2)

        c = chr(cv2.waitKey(0))

        if c == 's':

            if len(points) > 0:
                cv2.fillPoly(img_1, [np.array(points)], (0, 0, 255))
                cv2.fillPoly(img_2, [np.array(points)], (0, 0, 255))
                cv2.fillPoly(tiny_2015, [np.array(points)], (255, 255, 255))
                cv2.fillPoly(tiny_2017, [np.array(points)], (255, 255, 255))

        if c == '7':

            if len(points) > 0:
                cv2.fillPoly(img_2, [np.array(points)], (0, 0, 255))
                cv2.fillPoly(tiny_2017, [np.array(points)], (255, 255, 255))

        if c == '5':

            if len(points) > 0:
                cv2.fillPoly(img_1, [np.array(points)], (0, 0, 255))
                cv2.fillPoly(tiny_2015, [np.array(points)], (255, 255, 255))

    print(label_name_1, ' & ', label_name_2)
    cv2.imwrite(label_name_1, tiny_2015)
    cv2.imwrite(label_name_2, tiny_2017)

    return


if __name__ == '__main__':
    # for i in range(13,int(IM_ROWS // ROI_SIZE)+1):
    #     for j in range(int(IM_COLS // ROI_SIZE)):
    i = 13
    j = 20
    ss1 = '{}/2015/{}_{}_{}_.png'.format(DATA_DIR, i, j, ROI_SIZE)
    ss2 = '{}/2017/{}_{}_{}_.png'.format(DATA_DIR, i, j, ROI_SIZE)
    ss3 = '{}/mylabel_2015/{}_{}_{}_.png'.format(DATA_DIR, i, j, ROI_SIZE)
    ss4 = '{}/mylabel_2017/{}_{}_{}_.png'.format(DATA_DIR, i, j, ROI_SIZE)

    # if os.path.exists(ss3):
    #     continue
    # if os.path.exists(ss4):
    #     continue
    src_2015 = cv2.imread(ss1, 1)
    src_2017 = cv2.imread(ss2, 1)
    label_img(src_2015, src_2017, ss3, ss4)
