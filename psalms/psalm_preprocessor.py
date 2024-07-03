import pandas as pd
import numpy as np
import string
import re

import psalm_scraper

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer        # module for stemming

def process_psalm(psalm):
    
    punc_list = string.punctuation + '“' + '”' + '—'
    stopwords_english = stopwords.words('english') 
    
    # Instantiate stemming class
    stemmer = PorterStemmer() 

    tmp_ps = psalm.translate(str.maketrans('', '', punc_list)) #get psalm and remove punctuation
    ps = np.array(tmp_ps.lower().split(' ')) #force everything to lower case and split string.

    #removing stopwords
    clean_ps = []
    [clean_ps.append(j) for j in ps if j not in stopwords_english] #create array of tokenized words

    #stemming cleaned psalm
    ps_stem = []
    for word in clean_ps:
        stem_word = stemmer.stem(word)  # stemming word
        ps_stem.append(stem_word)
    
    if '' in ps_stem: ps_stem.remove('')
    return ps_stem

#Create vocabulary if desired
def create_raw_vocab(ps_dict):

    vocab = []
    for i in range(first, last+1):

        punc_list = string.punctuation + '“' + '”' + '—'

        tmp_ps = ps_dict[i].translate(str.maketrans('', '', punc_list)) #get psalm and remove punctuation
        ps = np.array(tmp_ps.lower().split(' ')) #split string and force to lower case. then create array of words from psalm to add to vocabulary
        [vocab.append(j) for j in ps if j not in vocab] #list comprehension to build vocabulary
    
    vocab.remove('')
    return vocab

#create clean corpus if desired
def clean_corpus(ps_dict, first=1, last=150):

    corpus = {}

    for i in range(first, last+1):
        ps_stem = process_psalm(ps_dict[i])

        corpus[i] = ps_stem
        
    return corpus

def build_freqs(psalm, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, p in zip(yslist, psalm):
        for word in process_psalm(p):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1    
    return freqs