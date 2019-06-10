import cv2 as cv
import numpy as np
import pandas as pd


def data2img(filename):
    data = pd.read_csv(filename, encoding='utf-8')
    num = len(data['id'])+1
    for i in range(num):
        filename = ' '
        img = data['feature'][i].split(' ')
        img_label = data['id'][i]
        filename = './img/'+str(img_label) + '/' + str(img_label)+'_' + str(i) + '.jpg'
        print("filename: ", filename)
        img = np.uint8(np.array(list(map(lambda x: int(x), img))).reshape(48, 48))
        print("num: ", i)
        cv.imwrite(filename, img)
        cv.imshow('1', img)
        cv.waitKey(1)
      #  cv.waitKey(0)


data2img('./data/train.csv')