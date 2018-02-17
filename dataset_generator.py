# -*- coding: utf-8 -*-
"""
COMP 551 A3

Author: Shatil Rahman
ID:  260606042

This module deals with preprocessing and generating the datasets from the yelp and IMDB review datasets

"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def genVocab(filename, output_filename, english):
    '''
    Generates the vocabulary from the filename given
    Inputs:
    filename - text file to read from
    output_filename - text file that will be created as the vocabulary
    english - 0 or 1, 0 meaning do not use stop_words for CountVectorizer
              and 1 to use 'english' as stop_words (words to ignore) for CountVectorizer
    '''
    text_file = open(filename, 'r')
    output_file = open(output_filename, 'w')
    
    entire_text = text_file.read()
    
    #initialize a CountVectorizer with no stop words, maximum 10000 words
    #in the vocabulary
    if(english):
        vectorizer = CountVectorizer(max_features=10000, stop_words='english',strip_accents='unicode')
    else:
        vectorizer = CountVectorizer(max_features=10000, strip_accents='unicode')
    
    
    X = vectorizer.fit_transform([entire_text])
    X = X.toarray()
    features = vectorizer.get_feature_names()
    
    vocabulary = []
    i = 0
    for feature in features:
        line = str(feature) + "\t" + str(i) + "\t" + str(X[0][i]) + str("\n")
        vocabulary.append(line)
        i = i + 1
    
    output_file.writelines(vocabulary)
    text_file.close()
    output_file.close()    
    return vocabulary    
    
    
#vocabulary = genVocab('hwk3_datasets/IMDB-train.txt', 'IMDB-vocab.txt',0)
#vocabulary = genVocab('hwk3_datasets/IMDB-train.txt', 'IMDB-vocab-processed.txt',1)

vocabulary_name = 'text_vocab.txt'
data_name = 'test.txt'
vocab = np.loadtxt(fname=vocabulary_name, dtype=str, delimiter='\t',usecols=(0,))
data_file = open(data_name, 'r')
lines = data_file.readlines()


vectorizer = CountVectorizer(vocabulary=vocab)

X = vectorizer.transform(lines)

X = X.toarray()

data_file.close()

    








