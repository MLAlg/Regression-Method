#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:07:50 2019

@author: samaneh
"""

import numpy as np
import matplotlib.pyplot as plt
# import database
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)
print(boston.feature_names)
print(np.max(boston.target), np.min(boston.target), np.mean(boston.target))
# split data and normalize data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=33)
from sklearn.preprocessing import StandardScaler
scalerx = StandardScaler().fit(x_train)
# scalery = StandardScaler().fit(y_train)
x_train = scalerx.transform(x_train)
# y_train = scalery.transform(y_train) ? y-dimension
x_test = scalerx.transform(x_test)
# y_test = scalery.transform(y_test)

from sklearn.model_selection import *
def train_and_evaluate(clf, x_train, y_train):
    clf.fit(x_train, y_train)
    print("Coefficient of determination on training set:", clf.score(x_train, y_train))    
    cv = KFold(5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, x_train, y_train, cv=cv)
    print("Average coefficient of determination using 5-fold cross validation:",  np.mean(scores))

# linear model: SGD
from sklearn import linear_model
clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty=None, random_state=42)
train_and_evaluate(clf_sgd, x_train, y_train)
print(clf_sgd.coef_) # the hyperplane coefficients

clf_sgd1 = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', random_state=42)
train_and_evaluate(clf_sgd1, x_train, y_train)

# svm for regression
from sklearn import svm
clf_svr = svm.SVR(kernel='linear')
train_and_evaluate(clf_svr, x_train, y_train)

clf_svr_poly = svm.SVR(kernel='poly')
train_and_evaluate(clf_svr_poly, x_train, y_train)

clf_svr_rbf = svm.SVR(kernel='rbf')
train_and_evaluate(clf_svr_rbf, x_train, y_train)

# Random Forest for regression-Extra Tree
from sklearn import ensemble
clf_et = ensemble.ExtraTreesRegressor(n_estimators=10, random_state=42)
train_and_evaluate(clf_et, x_train, y_train)
print(sorted(zip(clf_et.feature_importances_, boston.feature_names)))

# evaluation with the best performed model (Extra Tree)
from sklearn import metrics
def measure_performance(x, y, clf, show_accuracy=True, show_classification_report=True, 
    show_confusion_matrix=True, show_r2_score=False):
    y_pred = clf.predict(x)
    if show_accuracy: # ?
        print("Accuracy:{0: .3f}".format(metrics.accuracy_score(y, y_pred)), "\n")
    if show_classification_report: #?
        print("Classification report: ")
        print(metrics.classification_report(y, y_pred))
    if show_confusion_matrix: # ? 
        print("Confusion matrix: ")
        print(metrics.confusion_matrix(y, y_pred))
    if show_r2_score:
        print("Coefficient of determination:{0: .3f}".format(metrics.r2_score(y, y_pred)), "\n")

measure_performance(x_test, y_test, clf_et, show_accuracy=False, show_classification_report=False, show_confusion_matrix=False, show_r2_score=True)


































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    