# -*- coding: utf-8 -*-
"""
COMP 551 A3

Author: Shatil Rahman
ID:  260606042

This module trains the different classifiers, and evaluates their performances

"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
import inputBoW
import dataset_generator

dataset_generator.genData('hwk3_IMDB/IMDB-vocab.txt','testReviews.txt','testR.txt')

clf = BernoulliNB()
X_train, Y_train = inputBoW.loadBinBoW('hwk3_IMDB/IMDB-train.txt',15000,10000)
X_val, Y_val = inputBoW.loadBinBoW('testR.txt',2, 10000)

clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_val)
f1 = metrics.f1_score(Y_val, Y_pred)

print "F1 score for Naive Bayes is: " + str(f1)

