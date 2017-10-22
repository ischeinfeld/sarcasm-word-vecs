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
                        default="../data/labeled_data/positive/")
    parser.add_argument('-n', '--negative-tweets', dest='neg_dir',
                        help="Directory of example negative tweets",
                        default="../data/labeled_data/negative/")
    parser.add_argument('-c', '--sample-count', dest='sample_count',
                        help="Max number of samples of each sentiment",
                        default="1000000")

    args = parser.parse_args(arguments)

    # Create Tweets Iterators
    update("Creating tweet iterators...")
    pos_tweets_iter = Tweets([args.pos_dir])
    neg_tweets_iter = Tweets([args.neg_dir])
    update()

    # Save situtations to lists and shuffle
    update("Loading positive tweets...")
    pos_tweets = [' '.join(Tweets.filter_tags(tweet)) for tweet in pos_tweets_iter]
    update()

    update("Loading negative tweets...")
    neg_tweets = [' '.join(Tweets.filter_tags(tweet)) for tweet in neg_tweets_iter]
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

    # Build Pipeline
    print("Performing grid search...")
    pipeline = Pipeline([('vect', CountVectorizer()),
                         #('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])

    parameters = { #TODO check which parameters actually effect use in sarcasm detection
            'vect__tokenizer': [tokenizer],
            'vect__stop_words': [None],
            'vect__binary': [False],
            'vect__ngram_range': [(1,5)], 
            #'tfidf__norm': [None, 'l1', 'l2'],
            #'tfidf__use_idf': [True, False],
            #'tfidf__smooth_idf': [True, False],
            #'tfidf__sublinear_tf': [True, False],
            'clf__alpha': [1.0], # check range, these are guesses
            'clf__fit_prior': [False], # not sure what the distribution in sarcasm data is
    }

    clf_pipe = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    clf_pipe.fit(tweets[train], labels[train])
    print("Done in %0.3fs" % (time() - t0))
    print()

    # Print grid search results
    print("Best score: %0.3f" % clf_pipe.best_score_)
    print("Best parameters set:")
    best_parameters = clf_pipe.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print()

    # Evaluate classifier
    vect  = clf_pipe.best_estimator_.named_steps['vect']
    #tfidf = clf_pipe.best_estimator_.named_steps['tfidf']
    clf   = clf_pipe.best_estimator_.named_steps['clf']
    predicted = clf_pipe.predict(tweets[test])

    print("Classifier Evaluation:")
    print(metrics.classification_report(labels[test], predicted,
                                        target_names=["-", "+"]))

    # save classifier
    pickle.dump(clf_pipe, open(MODEL_FNAME, 'wb'))
    

def update(message="DONE.\n"):
    print(message, end='', flush=True)

def tokenizer(text):
    return text.split(' ')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
