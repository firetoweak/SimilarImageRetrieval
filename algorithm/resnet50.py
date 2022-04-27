# -*- coding:utf-8 -*-
# @Time     :2022/4/1 0:05
# @Author   :LiuHao
# @File     :resnet50.py
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


class Resnet50:
    def __init__(self, alg, samplePath=None, testPath=None):
        self.samplePath = samplePath
        self.testPath = testPath
        self.algName = alg

    def similarity(self):
        resnet = models.resnet50(pretrained=True)
        resnet.eval()
        transform = T.Compose([T.Resize(256), T.ToTensor()])
        image1 = transform(Image.open(self.samplePath, ).convert('RGB')).unsqueeze(0)
        feature1 = resnet(image1).detach().numpy()
        image2 = transform(Image.open(self.testPath, ).convert('RGB')).unsqueeze(0)
        feature2 = resnet(image2).detach().numpy()

        dot = np.sum(np.multiply(feature1, feature2), axis=1)
        norm = np.linalg.norm(feature1, axis=1) * np.linalg.norm(feature2, axis=1)
        return dot / norm