import string
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def add_sentiment_scores(ps_dict):
    """Add VADER sentiment scores to each verse in ps_dict."""
    analyzer = SentimentIntensityAnalyzer()
    for psalm_num in ps_dict:
        for verse in ps_dict[psalm_num]:
            verse['sentiment'] = analyzer.polarity_scores(verse['text'])


def add_binary_labels(ps_dict, pos_threshold=0.1, neg_threshold=-0.1):
    """Add binary sentiment labels based on VADER compound scores.
    Labels: 1 (positive), 0 (negative), None (neutral/excluded).
    """
    for psalm_num in ps_dict:
        for verse in ps_dict[psalm_num]:
            compound = verse['sentiment']['compound']
            verse['label'] = 1 if compound > pos_threshold else (0 if compound < neg_threshold else None)


def process_psalm(psalm):
    """Tokenize, remove stopwords, and stem a psalm verse."""
    punc_list = string.punctuation + '\u201c' + '\u201d' + '\u2014'
    stopwords_english = stopwords.words('english')
    stemmer = PorterStemmer()

    tmp_ps = psalm.translate(str.maketrans('', '', punc_list)).lower().split()
    clean_ps = [j for j in tmp_ps if j not in stopwords_english]
    ps_stem = [stemmer.stem(word) for word in clean_ps if word]
    if '' in ps_stem:
        ps_stem.remove('')
    return ps_stem


def build_freqs(psalms, ys):
    """Build a (word, label) frequency dictionary from a list of psalms and labels."""
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, p in zip(yslist, psalms):
        for word in process_psalm(p):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs


def naive_bayes_predict(psalm, logprior, loglikelihood):
    """Predict sentiment using Naive Bayes."""
    word_l = process_psalm(psalm)
    p = logprior
    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]
    return p
