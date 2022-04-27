# -*- coding:utf-8 -*-
# @Time     :2022/3/31 18:36
# @Author   :LiuHao
# @File     :md5.py

import hashlib


class Md5:
    def __init__(self, alg, samplePath=None, testPath=None):
        self.samplePath = samplePath
        self.testPath = testPath
        self.algName = alg

    def getMd5(self, str):
        md = hashlib.md5()
        md.update(str)
        dist = md.hexdigest()
        return dist

    def similarity(self):
        str1 = open(self.samplePath, 'rb')
        str2 = open(self.testPath, 'rb')
        md5Code1 = self.getMd5(str1.read())
        md5Code2 = self.getMd5(str2.read())
        str1.close()
        str2.close()
        return md5Code1 == md5Code2