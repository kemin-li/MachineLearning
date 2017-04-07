# -*- coding: utf-8 -*-
"""
Codes to load data from the book <MACHINE LEARNING IN ACTION>
Mostly copied from book with minor modifications.
Packed data into a dictionary MLIAdata and save it into a pickle file.
@author: K Li
Created on Sun Jan  8 2017
"""
import numpy as np
import pandas as pd
import pickle
ROOT_DIR = "D:/Projects/python_scripts/MachineLearningInAction"

MLIAdata = {}   # save test data from all chapters and later pickle.dump

### Chapter 2 k-Nearest-Neighbor -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch02
def file2matrix(filename):
    fr = open(filename)
    linesRead = fr.readlines()
    numberOfLines = len(linesRead)         #get the number of lines in the file
    dataMat = np.zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr.close()
    index = 0
    for line in linesRead:
        line = line.strip()
        listFromLine = line.split('\t')
        dataMat[index,:] = listFromLine[0:3]
        """classLabelVector.append(listFromLine[-1])
        index += 1
    labelMap = {}
    labelMap = {val:key for key,val in enumerate(set(classLabelVector))}
    return returnMat,classLabelVector, labelMap"""
    
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
        classLabelAeeay = np.array(classLabelVector)
    return dataMat,classLabelVector

filename = ROOT_DIR+"/Ch02/datingTestSet2.txt"
print("Loading data from...\n\t{}".format(filename))
ch2Data, ch2Label = file2matrix(filename)
MLIAdata['ch2']=(ch2Data, ch2Label)


### Chapter 3 Decision Trees -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch03
filename = ROOT_DIR+"/Ch03/lenses.txt"
print("Loading data from...\n\t{}".format(filename))
# processing contact lens data
with open(filename,'r') as fr:
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]

lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
#lensesTree = createTree(lenses,lensesLabels)
dataArray = np.array(lenses)
ch3Data = dataArray[:,:-1]
ch3Label = dataArray[:,-1]
MLIAdata['ch3']=(ch3Data, ch3Label)


### Chapter 4 Naive Bayes -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch04
dirname = ROOT_DIR+"/Ch04/email/"
print("Loading data from directory \n\t{}".format(dirname))
# Load email (spam/ham)
def loadEmail(filename):
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = open(dirname+'spam/%d.txt' % i, 'r').read()
        docList.append(wordList)
        classList.append(1)
        wordList = open(dirname+'ham/%d.txt' % i,'r').read()
        docList.append(wordList)
        classList.append(0)
    return docList, classList

textData, labelList = loadEmail(dirname)

# Convert email text to number
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
count_vect = CountVectorizer(token_pattern=r'\b\w\w+\b')
ch4Sparse = count_vect.fit_transform(textData)
ch4Data = ch4Sparse.toarray()
ch4Data = TfidfTransformer(use_idf=False).fit_transform(ch4Data)
ch4Label = np.array(labelList)
MLIAdata['ch4']=(ch4Data,ch4Label)
# check vacabulary 
#count_vect.get_feature_names()


### Chapter 5 Logistic Regression -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch05
dirname = ROOT_DIR+"/Ch05/"
print("Loading data from directory \n\t{}".format(dirname))
trainingFile = dirname+'horseColicTraining.txt'
trainingData = np.array(pd.read_csv(trainingFile, sep='\t',header=None))
testFile = dirname+'horseColicTest.txt'
testData = np.array(pd.read_csv(testFile, sep='\t',header=None))
mergedData = np.concatenate((trainingData,testData))
ch5Data = mergedData[:,:-1]
ch5Label = mergedData[:,-1]
MLIAdata['ch5']=(ch5Data,ch5Label)


### Chapter 6 Support vector machines -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch06
dirname = ROOT_DIR+"/Ch06/"
print("Loading data from directory \n\t{}".format(dirname))
def loadImages(dirName):
    from os import listdir
    def img2vector(filename):
        returnVect = np.zeros((1,1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
        fr.close()
        return returnVect

    hwLabels = []
    dataFileList = listdir(dirName)           #load the training set
    m = len(dataFileList)
    datagMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = dataFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        datagMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return datagMat, np.array(hwLabels)

ch6Data, ch6Label = loadImages(dirname+"trainingDigits")
ch6TestData, ch6TestLabel = loadImages(dirname+"testDigits")
MLIAdata['ch6']=(ch6Data, ch6Label, ch6TestData, ch6TestLabel)


### Chapter 7 Adaptive Boosting -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch07
def loadColicData2(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return np.array(dataMat), np.array(labelMat)

dirname = ROOT_DIR+"/Ch07/"
print("Loading data from directory \n\t{}".format(dirname))
ch7Data, ch7Label = loadColicData2(dirname+'horseColicTraining2.txt')
ch7TestData, ch7TestLabel = loadColicData2(dirname+'horseColicTest2.txt')
MLIAdata['ch7']=(ch7Data, ch7Label, ch7TestData, ch7TestLabel)


### Chapter 8 Regression -- source code and data
# Linear regression (LR), Locally weighted LR, ridge regression
# Lasso regression approximated by stagewise regression
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch08
def loadAbalone(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    fr.close()
    return np.array(dataMat), np.array(labelMat)
filename = ROOT_DIR+"/Ch08/abalone.txt"
ch8Data, ch8Label = loadAbalone(filename)
MLIAdata['ch8']=(ch8Data, ch8Label)


### Chapter 9 Tree based regression-- source code and data
# CART (Classification And Regression Trees)
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch09
def loadBikeData(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat
dirname = ROOT_DIR+"/Ch09/"
print("Loading data from directory \n\t{}".format(dirname))
trainData=np.array(loadBikeData(dirname+'bikeSpeedVsIq_train.txt'))
testData=np.array(loadBikeData(dirname+'bikeSpeedVsIq_test.txt'))
ch9Data, ch9Label = trainData[:,0], trainData[:,1]
ch9TestData, ch9TestLabel = testData[:,0], testData[:,1]
MLIAdata['ch9']=(ch9Data, ch9Label, ch9TestData, ch9TestLabel)


### Chapter 10 k-Means Clustering -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch10
dirname = ROOT_DIR+"/Ch10/"
print("Loading data from directory \n\t{}".format(dirname))
dataFile = dirname+'testSet.txt'
ch10Data = np.array(pd.read_csv(dataFile, sep='\t',header=None))
MLIAdata['ch10']=ch10Data

pickle.dump(MLIAdata, open('MLinActionData.pkl','wb'))
