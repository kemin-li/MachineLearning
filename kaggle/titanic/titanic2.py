# -*- coding: utf-8 -*-
"""
Created on Mon May 31th, 2017

To predict/classify survival of titanic passenger
Basic preprocessing for data, no thoughtful engineering
1. remove not relavant features, with too many missing values
2. filled missing 'Embarked' with mode, fill missing 'Age' with median values
grouped by sex and cabin (maybe later)
3. combine 'SibSp' and 'Parch' to a 'FamilySize' feature, cut into
solo/normal/big
4. Extract title from name, use what is common for both train and test
['NameTitle.', 'Mrs.', 'Miss.', 'Master.', 'Ms.', 'Col.', 'Rev.', 'Dr.']
Also add NameLength feature
5. Ticket # length and first letter
6. convert categorical values of 'Plass','Embarked','Sex','NameTitle',
'TitleLetter' to dummy variables
(or one hot code)
@author: K Li
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
#from sklearn.metrics import accuracy_score, log_loss

#dictionary to save training accuracy of various models
accModels = {}
predictions=pd.DataFrame()
seed = 2017015
submit = "submission2"
out2File = False

def feature_engineering(X_data):
    #from sklearn.preprocessing import OneHotEncoder
    # Processing missing values
    # Age has some missing values, filled with median value
    nRows, nCols = X_data.shape
    medianAge = X_data['Age'].median()
    ageFill = X_data.groupby(['Pclass','Sex'])['Age'].median()
    for row in range(nRows):
        if pd.isnull(X_data.loc[row,'Age']):
            X_data.loc[row,'Age'] = \
            ageFill[X_data.loc[row,'Pclass']][X_data.loc[row,'Sex']]

    X_data['Age'].fillna(medianAge, inplace=True)
    
    #Embarded has 2 missing values, filled with most frequent value 'S'
    X_data["Embarked"].fillna('S',inplace=True)
    # in test data, Fare column is missing one value
    medianFare = X_data['Fare'].median()
    X_data['Fare'].fillna(medianFare, inplace=True)

    # Processing outliers
    pass
    
    # feature engineering
    # Name, use length and title
    X_data['NameLen'] = X_data['Name'].apply(lambda x: len(x))
    get_last = lambda x: x.split(',')[1]
    get_title = lambda x: x.split()[0]
    X_data['NameTitle'] = X_data['Name'].apply(get_last).apply(get_title)
    # note title seen in training data and title seen in test sets may not 
    # exactly match. Therefore one may need to extract title from both
    # training data and test data
    # Here I manually enter common title in both train and test
    TitleSet = ['NameTitle.', 'Mrs.', 'Miss.', 'Master.', 'Ms.', 'Col.', 'Rev.', 'Dr.']
    X_data['NameTitle'] = np.where((X_data['NameTitle']).isin(TitleSet),
          X_data['NameTitle'],'Others')
    X_data.drop(['Name'], axis=1, inplace=True)
    
    # Create family size
    X_data['FamilySize'] = X_data['SibSp']+X_data['Parch']+1
    X_data['FamilySize'] = np.where(X_data['FamilySize'] == 1,'Solo',
          np.where(X_data['FamilySize'] == 1,'Normal','Big'))
    X_data.drop(['SibSp','Parch'], axis=1, inplace=True)
    
    # Ticket
    X_data['TicketLett'] = X_data['Ticket'].apply(lambda x: str(x)[0])
    X_data['TicketLett'] = X_data['TicketLett'].apply(lambda x: str(x))
    X_data['TicketLett'] = np.where((X_data['TicketLett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']),
          X_data['TicketLett'],
          np.where((X_data['TicketLett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                   'Low_ticket', 'Other_ticket'))
    X_data['TicketLen'] = X_data['Ticket'].apply(lambda x:len(x))
    X_data.drop(['Ticket'], axis=1, inplace=True)
    
    
    """# Use age band? not sure
    ageBands = 4
    ageBandSize = 16
    MAX_AGE = ageBands*ageBandSize
    X_data.loc[X_data.Age>=MAX_AGE,'Age'] = MAX_AGE
    # map age to respective age band
    for i in range(ageBands):
        X_data.loc[ (X_data.Age>i*ageBandSize) & (X_data.Age<=(i+1)*ageBandSize),
                   'AgeBand'] = i
    """

    # Categorical values to dummy variables
    # One-hot encoding with pandas.get_dummies
    cols = ['Sex', 'Pclass','Embarked','TicketLett','NameTitle','FamilySize']
    X_data = pd.get_dummies(X_data, columns=cols)
    '''features = X_data['Sex']
    fSet = set(features)
    genderDict = {f:i for i, f in enumerate(fSet)}
    X_data['Sex'] =  X_data['Sex'].map(genderDict)
    
    features = X_data['Embarked']
    fSet = set(features)
    embarkedDict = {f:i for i, f in enumerate(fSet)}
    X_data['Embarked'] =  X_data['Embarked'].map(embarkedDict)'''

    #Columns to be dropped
    #Cabin has too many missing values, delete columns
    #Name maybe related to social status, but drop it for now
    #Ticket is also hard to interpret
    #Sibsp/Parch are combined into FamilySize
    dropColumns = ['Cabin','PassengerId']
    X_data.drop(dropColumns, axis=1, inplace=True)
    return X_data

def majorityVote(predData):
    return predData.mode(axis=1)

def checkMissing(data):
    """Check for missing values
    Output column name, # of missing and percentage
    """
    N, M = data.shape
    columns = data.columns
    for col in columns:
        nMissing = data[col].isnull().sum()
        if nMissing:
            print("{} has {:d} missing values, {:.2f}%".format(col, nMissing, nMissing/N*100))
    return

def checkOutlier(data):
    pass
    return

def LRpredictor(X_train, y_train, X_test):
    '''Logistic Regression Classifier
    Input traning data ,target, and test data
    Output prabability of each label for test data'''
    from sklearn.linear_model import LogisticRegression as LR
    
    # Cross validation may not be needed for random forest classifier
    model = LR(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    logLoss = metrics.log_loss(y_train, y_pred)
    
    y_pred = model.predict(X_test)
    modelName = model.__class__.__name__
    accModels[modelName] = accuracy
    predictions[modelName] = y_pred

    return y_pred, accuracy

def KNNpredictor(X_train, y_train, X_test):
    '''Logistic Regression Classifier
    Input traning data ,target, and test data
    Output prabability of each label for test data'''
    from sklearn.neighbors import KNeighborsClassifier as KNN
    
    # Cross validation may not be needed for random forest classifier
    model = KNN()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    logLoss = metrics.log_loss(y_train, y_pred)
    
    y_pred = model.predict(X_test)
    modelName = model.__class__.__name__
    accModels[modelName] = accuracy
    predictions[modelName] = y_pred

    return y_pred, accuracy

def LRCVpredictor(X_train, y_train, X_test):
    '''Logistic Regression Classifier
    Input traning data ,target, and test data
    Output prabability of each label for test data'''
    from sklearn.linear_model import LogisticRegressionCV as LRCV
    
    # Cross validation may not be needed for random forest classifier
    model = LRCV(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    logLoss = metrics.log_loss(y_train, y_pred)
    
    y_pred = model.predict(X_test)
    modelName = model.__class__.__name__
    accModels[modelName] = accuracy
    predictions[modelName] = y_pred

    return y_pred, accuracy

def DTpredictor(X_train, y_train, X_test):
    '''Logistic Regression Classifier
    Input traning data ,target, and test data
    Output prabability of each label for test data'''
    from sklearn.tree import DecisionTreeClassifier as DT
    from sklearn.model_selection import StratifiedShuffleSplit as SSS
    
    # cross validation using StratifiedShuffleSplit
    sss = SSS(n_splits=5, test_size=0.2, random_state=seed)
    sss.get_n_splits(X_train, y_train)
    accuracy, logLoss, count = 0, 0, 0
    for train_ind, test_ind in sss.split(X_train, y_train):
        Xtrain, Xtest = X_train.iloc[train_ind], X_train.iloc[test_ind]
        ytrain, ytest = y_train[train_ind], y_train[test_ind]
        model = DT(random_state=1)
        model.fit(Xtrain, ytrain)
        y_pred = model.predict(Xtest)
        accuracy += metrics.accuracy_score(ytest, y_pred)
        logLoss += metrics.log_loss(ytest, y_pred)
        count += 1

    y_pred = model.predict(X_test)
    modelName = model.__class__.__name__
    accModels[modelName] = accuracy/count
    predictions[modelName] = y_pred

    return y_pred, accuracy

def RFCpredictor(X_train, y_train, X_test):
    ''' Input traning data ,target, and test data
    Output prabability of each label for test data'''
    from sklearn.ensemble import RandomForestClassifier as RFC
    
    # Cross validation may not be needed for random forest classifier
    model = RFC(n_estimators=25, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    logLoss = metrics.log_loss(y_train, y_pred)
    
    y_pred = model.predict(X_test)
    modelName = model.__class__.__name__
    accModels[modelName] = accuracy
    predictions[modelName] = y_pred

    return y_pred, accuracy

def RFCVpredictor(X_train, y_train, X_test):
    ''' Input traning data ,target, and test data
    Output prabability of each label for test data'''
    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn.model_selection import GridSearchCV
    
    # perform grid search and cross validation using GridSearchCV
    model = RFC(criterion='entropy', oob_score=True, random_state=seed)
    param_grid={'n_estimators':[40,50,60],
                "min_samples_split" : [2, 4, 8, 16]}
    
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    model = RFC(criterion='entropy',
                n_estimators=best_parameters['n_estimators'],
                min_samples_split=best_parameters["min_samples_split"],
                oob_score=True,
                random_state=seed)
    model.fit(X_train, y_train)
    #print(model.oob_score_)
    y_pred = model.predict(X_train)
    accuracy = model.oob_score_     #metrics.accuracy_score(y_train, y_pred)
    logLoss = metrics.log_loss(y_train, y_pred)
    
    y_pred = model.predict(X_test)
    modelName = model.__class__.__name__
    accModels[modelName] = accuracy
    predictions[modelName] = y_pred

    return y_pred, accuracy

def AdaBoostpredictor(X_train, y_train, X_test):
    ''' Input traning data ,target, and test data
    Output prabability of each label for test data'''
    from sklearn.ensemble import AdaBoostClassifier as ABC
    
    # Cross validation may not be needed for random forest classifier
    model = ABC(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    logLoss = metrics.log_loss(y_train, y_pred)
    
    y_pred = model.predict(X_test)
    modelName = model.__class__.__name__
    accModels[modelName] = accuracy
    predictions[modelName] = y_pred

    return y_pred, accuracy

def GBCpredictor(X_train, y_train, X_test):
    ''' Input traning data ,target, and test data
    Output prabability of each label for test data'''
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    from sklearn.model_selection import GridSearchCV
    
    # Cross validation may not be needed for random forest classifier
    model = GBC(random_state=seed)
    param_grid={"n_estimators": [50,100,200],
                "learning_rate":[0.05, 0.1, 0.2],
                "max_depth":    [2,3,4],
                "subsample":    [0.5,1],
                "min_samples_split": [2, 4]}
    
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_parameters = grid_search.best_estimator_.get_params()
    print(best_parameters)
    
    model = GBC(n_estimators=best_parameters['n_estimators'],
                learning_rate=best_parameters['learning_rate'],
                max_depth=best_parameters['max_depth'],
                min_samples_split=best_parameters["min_samples_split"],
                random_state=seed)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    logLoss = metrics.log_loss(y_train, y_pred)
    
    y_pred = model.predict(X_test)
    modelName = model.__class__.__name__
    accModels[modelName] = accuracy
    predictions[modelName] = y_pred

    return y_pred, accuracy

def SVMpredictor(X_train, y_train, X_test):
    ''' Input traning data ,target, and test data
    Output prabability of each label for test data'''
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    # normalize training data
    X_train_mean = X_train.mean(axis = 0)
    X_train_std = X_train.std(axis = 0)
    X_train_norm = (X_train - X_train_mean)/X_train_std
    
    X_test_mean = X_test.mean(axis = 0)
    X_test_std = X_test.std(axis = 0)
    X_test_norm = (X_test - X_test_mean)/X_test_std
    
    # Cross validation and hyper parameters selection for SVM
    model = SVC(kernel='rbf',random_state=seed)
    param_grid={'C':[1e-1, 1, 10, 100, 1000],
                'gamma':[0.01, 0.001, 0.0001]}
    
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train_norm, y_train)
    best_parameters = grid_search.best_estimator_.get_params()
    #print(best_parameters)
    
    model = SVC(kernel='rbf', random_state=seed,
                C=best_parameters['C'], gamma=best_parameters['gamma'])
    model.fit(X_train_norm, y_train)
    print("    Number of support vectors for SVC:", len(model.support_vectors_))
    y_pred = model.predict(X_train_norm)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    logLoss = metrics.log_loss(y_train, y_pred)
    
    y_pred = model.predict(X_test_norm)
    modelName = model.__class__.__name__
    accModels[modelName] = accuracy
    predictions[modelName] = y_pred

    return y_pred, accuracy

def LinearSVMpredictor(X_train, y_train, X_test):
    ''' Input traning data ,target, and test data
    Output prabability of each label for test data'''
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import GridSearchCV

    # normalize training data
    X_train_mean = X_train.mean(axis = 0)
    X_train_std = X_train.std(axis = 0)
    X_train_norm = (X_train - X_train_mean)/X_train_std
    
    X_test_mean = X_test.mean(axis = 0)
    X_test_std = X_test.std(axis = 0)
    X_test_norm = (X_test - X_test_mean)/X_test_std
    
    # Cross validation and hyper parameters selection for SVM
    model = LinearSVC(random_state=seed)
    param_grid={'C':[1e-2, 1e-1, 1, 10, 100],
                'tol':[0.01, 0.001, 0.0001]}
    
    grid_search = GridSearchCV(model, param_grid)
    grid_search.fit(X_train_norm, y_train)
    best_parameters = grid_search.best_estimator_.get_params()
    #print(best_parameters)
    
    model = LinearSVC(C=best_parameters['C'], tol=best_parameters['tol'],
                      random_state=seed)
    model.fit(X_train_norm, y_train)
    print("    LinearSVC coefficients:", model.coef_)
    y_pred = model.predict(X_train_norm)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    logLoss = metrics.log_loss(y_train, y_pred)
    
    y_pred = model.predict(X_test_norm)
    modelName = model.__class__.__name__
    accModels[modelName] = accuracy
    predictions[modelName] = y_pred

    return y_pred, accuracy

def MNBpredictor(X_train, y_train, X_test):
    ''' Input traning data ,target, and test data
    Output prabability of each label for test data'''
    from sklearn.naive_bayes import MultinomialNB as MNB
    # Cross validation may not be needed for random forest classifier
    model = MNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, y_pred)
    logLoss = metrics.log_loss(y_train, y_pred)
    
    y_pred = model.predict(X_test)
    modelName = model.__class__.__name__
    accModels[modelName] = accuracy
    predictions[modelName] = y_pred

    return y_pred, accuracy

train = pd.read_csv("train.csv")
fullColumns = train.columns
# check for general information on data
train.shape
train.head(3)
train.describe()
train.info()
checkMissing(train)
#train.set_index('PassengerId', inplace=True)
"""
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)

Data preprocessing steps
1. Check for missing values and how to deal with it. For example, fill
with certain values (0/mean/median), or remove some rows or cols
2. Check for outlier value, fill with certain values, or remove some
rows or cols
3. Conversion, it may be necessary to convert object/string/text data
into numeric data
4. Decide what features will be kept, experience or statistical methods
"""

X_train = feature_engineering(train)
y_train = X_train['Survived'].as_matrix()
X_train.drop('Survived', axis=1, inplace=True)


test = pd.read_csv("test.csv")
test_id = test["PassengerId"]
X_test = feature_engineering(test)

#DT_pred, accuracy = DTpredictor(X_train, y_train, X_test)
#print('Decision Tree training accuracy is %.4f'%accuracy)

KNN_pred, accuracy = KNNpredictor(X_train, y_train, X_test)
print('KNN training accuracy is %.4f'%accuracy)

LRCV_pred, accuracy = LRCVpredictor(X_train, y_train, X_test)
print('Logistic Regression CV training accuracy is %.4f'%accuracy)

#MNB_pred, accuracy = MNBpredictor(X_train, y_train, X_test)
#print('Multinomial Bayes training accuracy is %.4f'%accuracy)

#RF_pred, accuracy = RFCpredictor(X_train, y_train, X_test)
#print('Random Forest training accuracy is %.4f'%accuracy)

RFCV_pred, accuracy = RFCVpredictor(X_train, y_train, X_test)
print('Random Forest CV training accuracy is %.4f'%accuracy)

Ada_pred, accuracy = AdaBoostpredictor(X_train, y_train, X_test)
print('AdaBoost training accuracy is %.4f'%accuracy)

GB_pred, accuracy = GBCpredictor(X_train, y_train, X_test)
print('Gradient Boost training accuracy is %.4f'%accuracy)

SVM_pred, accuracy = SVMpredictor(X_train, y_train, X_test)
print('Support Vector Machine training accuracy is %.4f'%accuracy)

LSVM_pred, accuracy = LinearSVMpredictor(X_train, y_train, X_test)
print('Linear SVM training accuracy is %.4f'%accuracy)

print(accModels)
# Prepare for submission
# choose random forest, gradient boost, and majority vote of all predictors
submission_RF = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": predictions["RandomForestClassifier"]})
filename = "RandomForest_%s.csv"%submit
if out2File:
    submission_RF.to_csv(filename, index=False)

submission_GB = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": predictions["GradientBoostingClassifier"]})
filename = "GradientBoost_%s.csv"%submit
if out2File:
    submission_GB.to_csv(filename, index=False)

submission_SVC = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": predictions["SVC"]})
filename = "SVC_%s.csv"%submit
if out2File:
    submission_SVC.to_csv(filename, index=False)

ensemble_pred = majorityVote(predictions)
submission_Ensemble = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": ensemble_pred.iloc[:,0]})
filename = "Ensemble_%s.csv"%submit
if out2File:
    submission_Ensemble.to_csv(filename, index=False)
# check for correlation
#train1.corr()

