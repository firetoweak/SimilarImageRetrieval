# -*- coding:utf-8 -*-
# @Time     :2022/3/19 15:02
# @Author   :LiuHao
# @File     :makeData.py


import cv2
import numpy as np
from PIL import Image
# from skimage import transform, data
import torchvision.transforms as transform
from torchvision.transforms import *
import os
import changeColor
from RandomErase import RandomErasing


# 平移图片
def move(path, val):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros(imgInfo, np.uint8)
    for i in range(height):
        for j in range(width - val):
            dst[i, j + val] = img[i, j]
    return dst


# 旋转图片
def rotate(path, angle):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    matRotate = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), angle, 1)
    dst = cv2.warpAffine(img, matRotate, (width, height))
    return dst


# 镜像
def mirror(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1, dst=None)
    return img


if __name__ == "__main__":

    originPath = '../data/originImage'
    lider = os.listdir(originPath)
    outpath = '../data/save'
    hue = np.linspace(0, 360, 15)
    i = 0
    for image in lider:
        path = os.path.join(originPath, image)
        basename, ext = os.path.splitext(path)
        img = Image.open(path).convert('RGB')


        moveImage = move(path, 100)
        moveImage = Image.fromarray(moveImage)
        moveImage.save('{}_move.jpg'.format(basename))

        rotateImage = rotate(path, i+5)
        rotateImage = Image.fromarray(rotateImage)
        rotateImage.save('{}_rotate.jpg'.format(basename))

        mirrorImage = mirror(path)
        mirrorImage = Image.fromarray(mirrorImage)
        mirrorImage.save('{}_mirror.jpg'.format(basename))
        
        
        img2 = changeColor.hueChange(img, hue[i] / 360.)
        out_name = '{}_hue{:03d}.jpg'.format(basename, int(hue[i]))
        img2.save(out_name)


        # img = Image.open('../data/testImage/animals1.jpg')
        img = img.resize((400, 300), Image.BILINEAR)
        transformer = transform.Compose([
            transform.ToTensor(),
            RandomErasing()
            ])
        img_t = transformer(img)
        image_PIL = transforms.ToPILImage()(img_t)
        image_PIL.save('{}_erase.jpg'.format(basename))

        i += 1
