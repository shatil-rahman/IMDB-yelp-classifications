# -*- coding: utf-8 -*-
"""
COMP 551 A3

Author: Shatil Rahman
ID:  260606042

This module trains the different classifiers, and evaluates their performances

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from scipy.sparse import vstack
import inputBoW

def tuning(X_train, Y_train, X_val, Y_val, classifier, params):
    '''
    Tunes hyperparameters by running the classifier trained using training data
    on the range of parameters given, and returns the parameters which give the
    best f1-score on test data
    '''
    #Combine training and validation into one set
    X = vstack([X_train,X_val])
    Y_train.extend(Y_val)
    Y = np.array(Y_train)
    
    #Mark the training-validation splits
    train_i = np.ones((X_train.shape[0],), dtype = int) * -1
    valid_i = np.zeros((X_val.shape[0],), dtype = int)
    split_fold = np.concatenate((train_i, valid_i))
    ps = PredefinedSplit(split_fold)
    
    param_search = GridSearchCV(classifier, params, scoring=metrics.make_scorer(metrics.f1_score, average='macro'), cv=ps, return_train_score=True)
    param_search.fit(X,Y)
    results = param_search.cv_results_
    best_params = param_search.best_params_
    
    #Plotting
    #test_scores = results.get('split0_test_score')
    #par_ranges = params.values()
    #plt.plot(par_ranges[0],test_scores,'r-')
    #plt.show
    
    
    return best_params, results

def randomClf(X,low,high):
    Y = np.random.randint(low,high,size=X.shape[0])
    return Y
    
def majority(Y_train, X_val):
    counts = np.bincount(Y_train)
    majority_class = np.argmax(counts)
    Y_pred = [majority_class] * X_val.shape[0]
    return Y_pred

#Load the datasets
X_train, Y_train = inputBoW.loadBinBoW('hwk3_IMDB/IMDB-train.txt',15000,10000)
X_val, Y_val = inputBoW.loadBinBoW('hwk3_IMDB/IMDB-valid.txt',10000, 10000)

#X_trainf, Y_trainf = inputBoW.loadFreqBoW('hwk3_IMDB/IMDB-train.txt',25000,10000)
#X_valf, Y_valf = inputBoW.loadFreqBoW('hwk3_IMDB/IMDB-valid.txt',10000, 10000)

############################ Dumb Classifiers ###############################
'''
#Random Classifier:
Y_pred = randomClf(X_val, 1, 6)
f1 = metrics.f1_score(Y_val, Y_pred, average='micro')

print "F1 score for random classifier is: " + str(f1)

#Majority Classifier:
Y_pred = majority(Y_train, X_val)
f1 = metrics.f1_score(Y_val, Y_pred, average='micro')

print "F1 score for majority classifier is: " + str(f1)


########################## Naive Bayes ######################################
#Naive Bayes:
#Best alpha for IMDB is 0.2
#Best alpha for yelp is 0.0093
clf = BernoulliNB(alpha=0.2)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_val)
f1 = metrics.f1_score(Y_val, Y_pred, average='macro')

print "F1 score for Bernoulli Naive Bayes is: " + str(f1)


#Gaussian Naive Bayes:
#clf = GaussianNB()
#clf.fit(X_trainf.toarray(),Y_trainf)
#Y_pred = clf.predict(X_valf.toarray())
#f1 = metrics.f1_score(Y_valf, Y_pred, average='macro')

#print "F1 score for Gaussian Naive Bayes is: " + str(f1)

########################### Support Vector Machines #########################
#Linear SVM, binary BoW:
#Best C for IMDB set is 0.01
#Best C for yelp set is 0.1365
clf_SVM = LinearSVC(C=0.1365,loss='hinge')
clf_SVM.fit(X_train, Y_train)
Y_pred = clf_SVM.predict(X_val)
f1 = metrics.f1_score(Y_val, Y_pred, average='macro')

print "F1 score for Linear SVM, binary BoW is: " + str(f1)


#Linear SVM, frequency BoW:
#Best C for IMDB set is 149.0
#Best C for yelp set is 289.5/698.3
clf_SVM = LinearSVC(C=149.0,loss='hinge', tol=1e-6)
clf_SVM.fit(X_trainf, Y_trainf)
Y_pred = clf_SVM.predict(X_valf)
f1 = metrics.f1_score(Y_valf, Y_pred, average='macro')

print "F1 score for Linear SVM, frequency BoW is: " + str(f1)

########################### Decision Trees    ################################
'''
#Decision Tree:
#Best params for IMDB are max_depth=11, min_samples_leaf=12
#Best params for yelp are max_depth=18, min_samples_leaf=32
clf_DT = DecisionTreeClassifier(max_depth=11, min_samples_leaf=12)
clf_DT.fit(X_train,Y_train)
Y_pred = clf_DT.predict(X_val)
f1 = metrics.f1_score(Y_val, Y_pred, average='macro')

print "F1 score for Decision Tree, Binary BoW is: " + str(f1)
'''
#Decision Tree:
#Best params for IMDB are max_depth=10, min_samples_leaf=14
#Best params for yelp are max_depth=18, min_samples_leaf=20
clf_DT = DecisionTreeClassifier(max_depth=10, min_samples_leaf=14)
clf_DT.fit(X_trainf,Y_trainf)
Y_pred = clf_DT.predict(X_valf)
f1 = metrics.f1_score(Y_valf, Y_pred, average='macro')

print "F1 score for Decision Tree, frequency BoW is: " + str(f1)
'''

########################## HyperParamater Tuning #############################
#params_NB = {'alpha':np.linspace(0.001,0.01,100)}
#params_SVM = {'loss': ['hinge'], 'C': np.linspace(0.001,1.0,60)}
#params_DT = {'max_depth': np.linspace(2,100,50,dtype=int), 'min_samples_leaf': np.linspace(2,150,50,dtype=int)}


#best_alpha, results = tuning(X_trainf, Y_trainf, X_valf, Y_valf, BernoulliNB(), params_NB)
#best_C, results = tuning(X_train, Y_train, X_val, Y_val, LinearSVC(), params_SVM)
#best_DTpars, results = tuning(X_trainf, Y_trainf, X_valf, Y_valf, DecisionTreeClassifier(), params_DT)
