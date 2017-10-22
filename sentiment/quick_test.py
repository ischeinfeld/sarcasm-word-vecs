#!/usr/bin/env python

import os
import sys
import argparse
import random
import numpy as np 
import pickle
from pprint import pprint
from itertools import islice
from time import time
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

from tweets import Tweets

MODEL_FNAME = 'clf_pipe.pkl'

def main(arguments):

#    print("load classifier pipeline")
#    clf_pipe = pickle.load(open(MODEL_FNAME, 'rb'))
#
#    # test classifier
#    print("load classifier pipeline")
#    vect  = clf_pipe.best_estimator_.named_steps['vect']
#    clf   = clf_pipe.best_estimator_.named_steps['clf']

#    phrases = ["absolutely amazing", "amazing", "absolutely horrible", "horrible"]
#    for phrase in phrases:
#        print("Classify {}         : {}".format(phrase, clf_pipe.predict([phrase])))
#        print("Classify prob {}    : {}".format(phrase, clf_pipe.predict_proba([phrase])))
#        print("Classify log prob {}: {}".format(phrase, clf_pipe.predict_log_proba([phrase])))
#        print()
    
    ngrams, sentiments = create_sentiment()
    phrases = ["happy", "sad", "happy to see", "sad to see", "gobbledygook"]

    for phrase in phrases:
        index = ngrams[phrase]
        sentiment = sentiments[index]
        print("sentiment of {} is {}".format(phrase, sentiment))

#    print("feature_log_prob_ of that ^^")
#    print(clf.feature_log_prob_[:,index])
#
#    print("feature_count_ of that ^^")
#    print(clf.feature_count_[:,index])



def create_sentiment():
    """Returns a dictionary of [ngram] => [index] and a vector of [index] => [sentiment]"""
    clf_pipe = pickle.load(open(MODEL_FNAME, 'rb'))

    vect = clf_pipe.best_estimator_.named_steps['vect']
    clf  = clf_pipe.best_estimator_.named_steps['clf']

    ngrams = vect.vocabulary_
    sentiments = clf.feature_count_[1,:] / np.sum(clf.feature_count_, axis=0)

    return ngrams, sentiments

def update(message="DONE.\n"):
    print(message, end='', flush=True)

def tokenizer(text):
    return text.split(' ')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
