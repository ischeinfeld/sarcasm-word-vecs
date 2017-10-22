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

    # Parse optional filename arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--positive-tweets', dest='pos_dir',
                        help="Directory of example positive tweets",
                        default="../../data/labeled_data/positive/")
    parser.add_argument('-n', '--negative-tweets', dest='neg_dir',
                        help="Directory of example negative tweets",
                        default="../../data/labeled_data/negative/")
    parser.add_argument('-c', '--sample-count', dest='sample_count',
                        help="Max number of samples of each sentiment",
                        default="10")

    args = parser.parse_args(arguments)

    # Create Tweets Iterators
    update("Creating tweet iterators...")
    pos_tweets_iter = Tweets([args.pos_dir])
    neg_tweets_iter = Tweets([args.neg_dir])
    update()

    # Save situtations to lists and shuffle
    update("Loading positive tweets...")
    pos_tweets = [' '.join(tweet) for tweet in pos_tweets_iter]
    update()

    update("Loading negative tweets...")
    neg_tweets = [' '.join(tweet) for tweet in neg_tweets_iter]
    update()

    update("Selecting balanced sample sets...")
    sample_count = int(args.sample_count)
    pos_tweets = resample(pos_tweets, n_samples=sample_count,
                              replace=False, random_state=1)
    neg_tweets = resample(neg_tweets, n_samples=sample_count,
                              replace=False, random_state=2)
    update()

    # Shuffle tweets and split into training, dev, and test
    update("Shuffle tweets and split into training, dev, and test sets...")
    pos_labels = [1 for _ in pos_tweets]
    neg_labels = [0 for _ in neg_tweets]

    tweets = np.append(pos_tweets, neg_tweets)
    labels = np.append(pos_labels, neg_labels)

    tweets, labels = shuffle(tweets, labels, random_state=2)
    size = len(labels)
    train = slice(0, int(0.8 * size))
    dev = slice(int(0.8 * size), int(0.9 * size))
    test = slice(int(0.8 * size), size - 1)
    update()
    print()
    clf_pipe = pickle.load(open(MODEL_FNAME, 'rb'))

    # Evaluate classifier
    vect  = clf_pipe.best_estimator_.named_steps['vect']
    clf   = clf_pipe.best_estimator_.named_steps['clf']
    predicted = clf_pipe.predict(tweets[test])

    print("Classifier Evaluation:")
    print(metrics.classification_report(labels[test], predicted,
                                        target_names=["-", "+"]))

    

def update(message="DONE.\n"):
    print(message, end='', flush=True)

def tokenizer(text):
    return text.split(' ')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
