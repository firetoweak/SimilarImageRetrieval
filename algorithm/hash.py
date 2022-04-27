# -*- coding:utf-8 -*-
# @Time     :2022/3/19 20:29
# @Author   :LiuHao
# @File     :hash.py

import cv2
import numpy as np
import time
import os


class Hash:
    def __init__(self, alg, samplePath=None, testPath=None):
        self.samplePath = samplePath
        self.testPath = testPath
        self.algName = alg

    def pHash(self, img, leng=32, wid=32):
        img = cv2.resize(img, (leng, wid))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(np.float32(gray))
        dct_roi = dct[0:8, 0:8]
        avreage = np.mean(dct_roi)
        phash_01 = (dct_roi > avreage) + 0
        phashList = phash_01.reshape(1, -1)[0].tolist()
        hash = ''.join([str(x) for x in phashList])
        return hash

    def dHash(self, img, leng=9, wid=8):
        img = cv2.resize(img, (leng, wid))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
        hash = []
        for i in range(wid):
            for j in range(wid):
                if image[i, j] > image[i, j + 1]:
                    hash.append(1)
                else:
                    hash.append(0)
        return hash

    def aHash(self, img, leng=8, wid=8):
        img = cv2.resize(img, (leng, wid))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avreage = np.mean(image)
        hash = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] >= avreage:
                    hash.append(1)
                else:
                    hash.append(0)
        return hash

    def similarity(self):
        num = 0
        hash1 = hash2 = 0
        image1 = cv2.imread(self.samplePath)
        image2 = cv2.imread(self.testPath)
        if self.algName == 'ahash':
            hash1 = self.aHash(image1)
            hash2 = self.aHash(image2)
        elif self.algName == 'phash':
            hash1 = self.pHash(image1)
            hash2 = self.pHash(image2)
        elif self.algName == 'dhash':
            hash1 = self.dHash(image1)
            hash2 = self.dHash(image2)
        for index in range(len(hash1)):
            if hash1[index] != hash2[index]:
                num += 1
        return 1 - num * 1.0 / 64