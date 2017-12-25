# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:17:28 2017
K-近邻算法
@author: 完颜幸幸
"""

import numpy as np
import operator# 运算符模块

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]# 读取dataSet的第一维度(即0轴）长度，使用shape[0]
    diffMat =np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
labels = ['A','A','B','B']

print(classify0([0,0], group, labels,3))