# -*- coding:utf-8 -*-
# @Time     :2022/3/19 20:30
# @Author   :LiuHao
# @File     :SIFT.py


import cv2
import os


"""
FLANN是类似最近邻的快速匹配库
    它会根据数据本身选择最合适的算法来处理数据
    比其他搜索算法快10倍
"""


class SIFT:
    def __init__(self, alg, samplePath=None, testPath=None):
        self.samplePath = samplePath
        self.testPath = testPath
        self.algName = alg

    # 创建sift检测器
    def sift(self, img1, img2):
        sift = cv2.xfeatures2d.SIFT_create()
        # 查找监测点和匹配符
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        """
        keypoint是检测到的特征点的列表
        descriptor是检测到特征的局部图像的列表
        """
        bf = cv2.BFMatcher()
        matches1 = bf.match(des1, des2)
        # 获取flann匹配器
        FLANN_INDEX_KDTREE = 1
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        searchParams = dict(checks=50)
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
        # 进行匹配
        matches = flann.knnMatch(des1, des2, k=2)

        return matches

    def similarity(self):
        # 构建匹配点，阈值0.7
        matchNumber = 0
        img1 = cv2.imread(self.samplePath)
        img2 = cv2.imread(self.testPath)
        matches = self.sift(img1, img2)
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.4 * n.distance:
                matchNumber += 1
        if matchNumber/len(matches) > 0.1:
            return 1
        return 0

"""
import matplotlib.pyplot as plt
path1 = '../data/originImages/image2.jpg'
path2 = '../data/testImages/image2_move.jpg'
img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

sift = cv2.xfeatures2d.SIFT_create()
# 查找监测点和匹配符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 获取flann匹配器
FLANN_INDEX_KDTREE = 1
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
# 进行匹配
matches = flann.knnMatch(des1, des2, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
numMatches = 0
for i, (m, n) in enumerate(matches):
    if m.distance < 0.4 * n.distance:
        matchesMask[i] = [1, 0]
        numMatches += 1
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.figure(figsize=(20, 20))
plt.imshow(img3, cmap='gray'), plt.title('Matched Result'), plt.axis('off')
plt.show()
ans = numMatches/len(matches)
print(ans)
i =0
"""


