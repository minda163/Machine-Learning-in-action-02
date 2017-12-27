# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:17:28 2017
K-近邻算法
@author: 完颜幸幸
"""

import numpy as np
import operator# 运算符模块
from os import listdir # 从os模块中导入函数listdir，它可以列出给定目录的文件名

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
        #print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
labels = ['A','A','B','B']

print(classify0([0,0], group, labels,3))

def file2matrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)          #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,3))  #prepare matrix to return
    classLabelVector = []                    #prepare labels return   
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')    
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat,classLabelVector

datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
#print(datingDataMat,datingLabels) #出错：为什么全是0:原因在fr.readlines()的直接调用上（不可直接多次调用）
#creating scatter plots with Matpotlib
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15*(np.array(map(int,datingLabels))),15*(np.array(map(int,datingLabels)))) #重点修改本行

tm = list(range(len(datingLabels)))

for i in range(len(datingLabels)):#把内容从字符串转化为数值并保存在列表tm中
    if datingLabels[i]=='didntLike':
        tm[i] = 1
    if datingLabels[i]=='smallDoses':
        tm[i] = 2
    if datingLabels[i]=='largeDoses':
        tm[i] = 3
tm = np.array(tm)    
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*tm,15.0*tm,linewidths=1,alpha=0.6)
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
'''
#画出按labels分颜色的图失败，原因是没有能正确的把datingLabels里的字符串转化为int或者float的格式。
#另外，就是python3与python2不兼容的问题：
#python2的map是得到的list，而python3的map得到的是object，我不会调用？需要学习。。。
加了一个for循环，也改变了文件，把label从字符编程了数字，还是有错，显示的图不是彩色的是黑白的。。
把list转化为array就变成彩色的了。
成功！
'''
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*tm,15.0*tm,linewidths=1,alpha=0.6)
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Frequent Flyier Miles Earned Per Year')
plt.show()

#Data-normalizing
def autoNorm(dataSet):
    minVals = dataSet.min(0)#0代表坐标方向（找出每列的最小值)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]#0代表坐标方向（得到数据的列数)
    #数值归一化公式（转化为0~1区间）：newValue = (oldValue-min)/(max-min)
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    #元素相除，在NumPy中矩阵相除要调用函数linalg。solve（matA，matB）
    return normDataSet,ranges,minVals
    
print(autoNorm(datingDataMat))

#Classifier testing code for dating site
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt') #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
#        print("the classifier came back with: %s, the real answer is: %s" %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)    
    
datingClassTest()

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
print(img2vector('testDigits/0_13.txt'))

#Handwritten digits testing code
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits') #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
#        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))
handwritingClassTest()