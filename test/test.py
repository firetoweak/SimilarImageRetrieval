# -*- coding:utf-8 -*-
# @Time     :2022/3/19 19:34
# @Author   :LiuHao
# @File     :test.py

import algorithm.hash as hash
import algorithm.hist as hist
import algorithm.SIFT as sift
import algorithm.ssim6 as ssim
import algorithm.resnet50 as resnet
import algorithm.md5 as md5
import os
import cv2
import time

sampleImagesPath = '../data/originImages'
testImagesPath = '../data/testImages'
negativePath = '../data/irrelevant'

# 制作测试输入
def dataLoad(sampleImagesPath, testImagesPath):
    samlpeListDir = os.listdir(sampleImagesPath)
    testListDir = os.listdir(testImagesPath)
    dataDir = {}
    for img in samlpeListDir:
        dataDir[img] = []
    for img in testListDir:
        basename = os.path.splitext(img)[0]
        basename, evt = basename.split('_')
        orginName = basename+'.jpg'
        if orginName in dataDir:
            if evt.find('erase') == -1:
                dataDir[orginName].append(img)
    return dataDir


# 测试算法
def test(sampleImagesPath, testImagesPath, negativePath, algroithm, w):
    dataDir = dataLoad(sampleImagesPath, testImagesPath)
    negativeDir = dataLoad(sampleImagesPath, negativePath)
    TP = FN = FP = 0
    start = time.time()
    for orginImage in dataDir:
        testImages = dataDir[orginImage]
        negativeImages = negativeDir[orginImage]
        sampleImage = os.path.join(sampleImagesPath, orginImage)
        for img in testImages:
            testImage = os.path.join(testImagesPath, img)
            alg = algroithm
            alg.samplePath = sampleImage
            alg.testPath = testImage
            if alg.similarity() > w:
                TP += 1
            else:
                FN += 1
        for img in negativeImages:
            negativeImage = os.path.join(negativePath, img)
            alg = algroithm
            alg.samplePath = sampleImage
            alg.testPath = negativeImage
            if alg.similarity() > w:
                FP += 1
    R = TP / (TP + FN)
    P = TP / (TP + FP)
    end = time.time()
    evTime = end - start
    print(algroithm.algName, '的召回率为:{:.3f}'.format(R), '精确率为：{:.3f}'.format(P), "执行时间为:{:.3f}".format(evTime))


if __name__ == '__main__':
    w = 0.7
    test(sampleImagesPath, testImagesPath, negativePath, md5.Md5('md5'), w)
    test(sampleImagesPath, testImagesPath, negativePath, hash.Hash('ahash'), w)
    test(sampleImagesPath, testImagesPath, negativePath, hash.Hash('dhash'), w)
    test(sampleImagesPath, testImagesPath, negativePath, hash.Hash('phash'), w)
    test(sampleImagesPath, testImagesPath, negativePath, hist.Hist('hist'), w)
    test(sampleImagesPath, testImagesPath, negativePath, hist.Hist('histRGB'), w)
    test(sampleImagesPath, testImagesPath, negativePath, sift.SIFT('sift'), w)
    test(sampleImagesPath, testImagesPath, negativePath, ssim.SSIM('ssimAlg'), w)
    test(sampleImagesPath, testImagesPath, negativePath, resnet.Resnet50('resNetAlg'), w)