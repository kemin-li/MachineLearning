# -*- coding: utf-8 -*-
"""
Implement sample codes from the book <MACHINE LEARNING IN ACTION> with
scikit-learn. Compare results whenever possible
Environment:
-- Python 3.5.2 |Anaconda 4.1.1 (64-bit)| [MSC v.1900 64 bit (AMD64)]
-- Scikit-learn 0.17.1
@author: K Li
Created on Sun Jan 8 2017
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# load data generated by MLinActionData.py
# data is a dict, {'ch2', data_tuple, 'ch3', data_tuple,...}
data = pickle.load(open('MLinActionData.pkl','rb'))
ch2Data, ch2Label = data['ch2']
ch3Data, ch3Label = data['ch3']
ch4Data, ch4Label = data['ch4']
ch5Data, ch5Label = data['ch5']
ch6Data, ch6Label, ch6TestData, ch6TestLabel = data['ch6']
ch7Data, ch7Label, ch7TestData, ch7TestLabel = data['ch7']
ch8Data, ch8Label = data['ch8']
ch9Data, ch9Label, ch9TestData, ch9TestLabel = data['ch9']
ch10Data = data['ch10']

def classfierResults(test, predict, classifier):
    '''Print accuracy of the classifier
    '''
    print('******************* %s ********************' % classifier)
    is_binary_class = (len(np.unique(test)) == 2)
    if is_binary_class:  
        precision = metrics.precision_score(test, predict)  
        recall = metrics.recall_score(test, predict)  
        print('precision: %.2f%%, recall: %.2f%%' % (100*precision, 100*recall))
    accuracy = metrics.accuracy_score(test, predict)  
    print('accuracy: %.2f%%' % (100 * accuracy))

def showPrediction(vTest, vPredict, modelName):
    '''Plot y_test(blue) and y_pred(red) in the same plot for comparison'''
    plt.figure()
    plt.plot(vTest,'b',vPredict, 'r')
    plt.title(modelName)

def plotClusters(dataSet, centerSet, clusterSet, modelName):
    featureSet = list(set(clusterSet))
    numFeatures = len(featureSet)
    colors = list('rgbcymk')
    markers = list('sod^v<>')
    K = len(markers)
    plt.figure()
    if(numFeatures>K):
        raise Exception('Number of features can not be greater than %d.'%K)
    for n in range(numFeatures):
        dataIndices = np.nonzero(clusterSet == featureSet[n])[0]
        nthCluster = dataSet[dataIndices,:]
        plt.scatter(nthCluster[:,0],nthCluster[:,1],s=50,c=colors[n],marker=markers[n])
        plt.scatter(centerSet[n,0],centerSet[n,1],s=200,linewidths=3,c='k',marker='+')
    plt.title(modelName)

### Chapter 2 k-Nearest-Neighbor -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch02
# prediction error in book (page 32) is 24%. Slightly different from results
# here because we use random split instead of fixed split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(
    ch2Data, ch2Label, test_size=0.20, random_state=42)
model = KNeighborsClassifier(3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
classfierResults(y_test, y_pred, 'kNN')


### Chapter 3 Decision Trees -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch03
# in order for decision tree classifier in scikit-learn to work, we need
# map string labels to integers
from sklearn.tree import DecisionTreeClassifier
def str2intMapping(Data, Labels):
    m,n = Data.shape
    #l = len(Labels)
    intData = np.zeros((m,n))
    intLabels = np.zeros(Labels.shape)
    mapList = []
    for colInd in range(n):
        colVec = Data[:,colInd]
        mapDict = {val:ind for ind,val in enumerate(np.unique(colVec))}
        mapList.append(mapDict)
        intData[:,colInd] = list(map(lambda x:mapDict[x], colVec))
    colVec =Labels
    mapDict = {val:ind for ind,val in enumerate(np.unique(colVec))}
    mapList.append(mapDict)
    intLabels = list(map(lambda x:mapDict[x], colVec))
    return intData, intLabels, mapList
    

# need to convert str-type feature values to integer
intData, intLabel, mapList = str2intMapping(ch3Data, ch3Label)
X_train, X_test, y_train, y_test = train_test_split(
    intData, intLabel, test_size=0.20, random_state=0)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
classfierResults(y_test, y_pred, 'Decision Tree')


### Chapter 4 Naive Bayes -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch04
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(
    ch4Data, ch4Label, test_size=0.20, random_state=0)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
classfierResults(y_test, y_pred, 'Naive Bayes')


### Chapter 5 Logistic Regression -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch05
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(
    ch5Data, ch5Label, test_size=0.20, random_state=0)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
classfierResults(y_test, y_pred, 'Logistic Regression')
print('Compared with 65% accuracy in the book')


### Chapter 6 Support vector machine -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch06
from sklearn.svm import SVC
(X_train, X_test, y_train, y_test) =(
    ch6Data, ch6TestData, ch6Label, ch6TestLabel)
#\gamma = 1/(2*\sigma^2)
#exp(-\gamma|x-y|^2)
model = SVC(kernel='rbf', tol=0.0001, gamma=0.005, probability=True)
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)
classfierResults(y_test, y_pred, 'Support Vector Machine')


### Chapter 7 Adaptive Boosting -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch07
from sklearn.ensemble import AdaBoostClassifier
(X_train, X_test, y_train, y_test) =(
    ch7Data, ch7TestData, ch7Label, ch7TestLabel)
model  = AdaBoostClassifier()
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)
classfierResults(y_test, y_pred, 'Adaptive Boosting')
print('Compared with 79% accuracy in the book')


### Chapter 8 Regression -- source code and data
# Linear regression (LR), Locally weighted LR, ridge regression
# Lasso regression approximated by stagewise regression
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch08
from sklearn.linear_model import Ridge, Lasso
X_train, X_test, y_train, y_test = train_test_split(
    ch8Data, ch8Label, test_size=0.20, random_state=0)
model = Ridge()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
MSE = np.linalg.norm(y_test-y_pred)/len(y_test)
print('Ridge Regression predictor MSE is %.2f' %MSE)
showPrediction(y_test, y_pred, 'Ridge regressor')

model = Lasso(alpha=0.02)
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)
MSE = np.linalg.norm(y_test-y_pred)/len(y_test)
print('Lasso Regression predictor MSE is %.2f' %MSE)
showPrediction(y_test, y_pred, 'Lasso regressor')


### Chapter 9 Tree based regression-- source code and data
# CART (Classification And Regression Trees)
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch09
from sklearn.tree import DecisionTreeRegressor
(X_train, X_test, y_train, y_test) =(
    ch9Data.reshape(-1,1), ch9TestData.reshape(-1,1), ch9Label, ch9TestLabel)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
MSE = np.linalg.norm(y_test-y_pred)/len(y_test)
print('Ridge Regression predictor MSE is %.2f' %MSE)
showPrediction(y_test, y_pred, 'DT regressor')


### Chapter 10 k-Means Clustering -- source code and data
# https://github.com/pbharrin/machinelearninginaction/tree/master/Ch10
from sklearn.cluster import KMeans, MiniBatchKMeans
model = KMeans(n_clusters = 4)
model.fit(ch10Data)
myCentroids, clustAssing = model.cluster_centers_,  model.labels_
plotClusters(ch10Data, myCentroids, clustAssing, 'k-Means')
