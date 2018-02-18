# -*- coding: utf-8 -*-
"""

COMP 551 A3

Author: Shatil Rahman
ID:  260606042

This module runs reads from the formatted and preprocessed data files, 
and returns them as Binary Bag of Words or Frequency Bag of Words forms in sparse
csr_matrices (and the Y_set target values)


"""

from scipy.sparse import csr_matrix
import numpy as np
from sklearn import preprocessing



def loadBinBoW(datafile_name, n_samples, n_features):

    data_file = open(datafile_name, 'r')
    
    Y_set = []
    
    examples = np.zeros((n_samples,n_features), dtype=int)
    
    i = 0
    for line in data_file:
        X,Y = line.split("\t")
        X = X.split()
        Y = int(Y)
        Y_set.append(Y)
            
        for ID in X:
            j = int(ID)
            examples[i][j] = 1
    
        i = i + 1
    
    X = csr_matrix(examples)
    return X, Y_set
    
def loadFreqBoW(datafile_name, n_samples, n_features):

    data_file = open(datafile_name, 'r')
        
    Y_set = []
    
    examples = np.zeros((n_samples,n_features), dtype=float)
    
    i = 0
    for line in data_file:
        X,Y = line.split("\t")
        X = X.split()
        Y = int(Y)
        Y_set.append(Y)
            
        for ID in X:
            j = int(ID)
            examples[i][j] = examples[i][j] + 1
    
        
        i = i + 1
    
    X = csr_matrix(examples)
    preprocessing.normalize(X, norm='l1',axis=1,copy=False)
    
    return X, Y_set




