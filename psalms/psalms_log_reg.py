import pandas as pd
import numpy as np
import string
import re
import random

import psalm_scraper as ps
import psalm_preprocessor as prp

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def extract_features(psalm, freqs, process_psalm=prp.process_psalm):
    '''
    Input: 
        tweet: a string containing one tweet
        freqs: a dictionary to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
        
    A function that extracts word counts based on positive/negative associations
    '''
    
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_psalm(psalm)
    
    # 3 elements for [bias, positive, negative] counts
    x = np.zeros(2) 
        
    # loop through each word in the list of words
    for word in word_l:
        
        # increment the word count for the positive label 1
        x[0] += freqs[(word,1)] if (word,1) in freqs else 0
        
        # increment the word count for the negative label 0
        x[1] += freqs[(word,0)] if (word,0) in freqs else 0
            
    x = x[None, :]  # adding batch dimension for further processing
    assert(x.shape == (1, 2))
    return x

def generate_covariates(psalms, freqs):
    X = np.zeros((len(psalms), 2))
    for p in range(len(psalms)):
        X[p, :] = extract_features(psalms[p], freqs)
        
        return X
    
def get_ps_labels():
    outcomes_df = pd.read_csv('psalms_sentiment.csv')
    outcomes = np.array(outcomes_df['positive'])
    
    return outcomes

def logistic_train_test_split(X, outcomes):
    X_train, X_test, y_train, y_test = train_test_split(X, outcomes, test_size=0.2)
    
    return X_train, X_test, y_train, y_test

def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    ### START CODE HERE ###
    
    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)
    
    # make the prediction using x and theta
    y_pred = sigmoid(x.dot(theta))
    
    ### END CODE HERE ###
    
    return y_pred
