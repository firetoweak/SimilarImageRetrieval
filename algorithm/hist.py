# -*- coding:utf-8 -*-
# @Time     :2022/3/22 14:41
# @Author   :LiuHao
# @File     :hist.py

import cv2
import os


class Hist:
    def __init__(self, alg, samplePath=None, testPath=None):
        self.samplePath = samplePath
        self.testPath = testPath
        self.algName = alg

    def calculate(self, image1, image2):
        # 灰度直方图算法
        # 计算单通道的直方图的相似值
        hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
        degree = 1 - cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_BHATTACHARYYA)
        return degree

    def classify_hist_with_split(self, image1, image2, size=(256, 256)):
        # RGB每个通道的直方图相似度
        # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
        image1 = cv2.resize(image1, size)
        image2 = cv2.resize(image2, size)
        sub_image1 = cv2.split(image1)
        sub_image2 = cv2.split(image2)
        sub_data = 0
        for im1, im2 in zip(sub_image1, sub_image2):
            sub_data += self.calculate(im1, im2)
        sub_data = sub_data / 3
        return sub_data

    def similarity(self):
        img1 = cv2.imread(self.samplePath)
        img2 = cv2.imread(self.testPath)
        if self.algName == 'hist':
            return self.calculate(img1, img2)
        return self.classify_hist_with_split(img1, img2)