# -*- coding:utf-8 -*-
# @Time     :2022/3/19 15:02
# @Author   :LiuHao
# @File     :makeData.py


import cv2
import numpy as np
from PIL import Image
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
    originPath = '../data/originImages'
    lider = os.listdir(originPath)
    outPath = '../data/changeImages'
    if os.path.exists(outPath) is False:
        os.makedirs(outPath)
    hue = np.linspace(0, 360, 15)
    i = 0
    for image in lider:
        path = os.path.join(originPath, image)
        basename, ext = os.path.splitext(path)
        imgName  = basename.split('\\')[-1]
        img = Image.open(path).convert('RGB')


        moveImage = move(path, 100)
        moveImage = Image.fromarray(moveImage)
        moveImage.save(outPath + '/{}_move.jpg'.format(imgName))

        rotateImage = rotate(path, i+5)
        rotateImage = Image.fromarray(rotateImage)
        rotateImage.save(outPath + '/{}_rotate.jpg'.format(imgName))

        mirrorImage = mirror(path)
        mirrorImage = Image.fromarray(mirrorImage)
        mirrorImage.save(outPath + '/{}_mirror.jpg'.format(imgName))

        img2 = changeColor.hueChange(img, hue[i] / 360.)
        out_name = outPath + '/{}_hue{:03d}.jpg'.format(imgName, int(hue[i]))
        img2.save(out_name)

        img = img.resize((400, 300), Image.BILINEAR)
        transformer = transform.Compose([
            transform.ToTensor(),
            RandomErasing()
            ])
        img_t = transformer(img)
        image_PIL = transforms.ToPILImage()(img_t)
        image_PIL.save(outPath + '/{}_erase.jpg'.format(imgName))

        i += 1
