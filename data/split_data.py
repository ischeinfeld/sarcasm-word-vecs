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

from tweets import Tweets

POS_DIR = "./labeled_data/positive/"
NEG_DIR = "./labeled_data/negative/"
SAR_DIR = "./labeled_data/sarcastic/"

SPLIT_DATA_DIR = "./split_data/"
RAND_SEED = 178

def main(pos_dir, neg_dir, sar_dir, random_seed):
    np.random.seed(random_seed)

    # Create tweets iterators
    update("Creating tweet iterators...")
    pos_tweets_iter = Tweets([pos_dir])
    neg_tweets_iter = Tweets([neg_dir])
    sar_tweets_iter = Tweets([sar_dir])
    update()

    # Save situtations to lists and shuffle
    update("Loading positive tweets...")
    pos_tweets = [' '.join(Tweets.filter_tags(tweet)) for tweet in pos_tweets_iter]
    pos_tweets = shuffle(pos_tweets)
    update()

    update("Loading negative tweets...")
    neg_tweets = [' '.join(Tweets.filter_tags(tweet)) for tweet in neg_tweets_iter]
    neg_tweets = shuffle(neg_tweets)
    update()

    update("Loading sarcastic tweets...")
    sar_tweets = [' '.join(Tweets.filter_tags(tweet)) for tweet in sar_tweets_iter]
    sar_tweets = shuffle(sar_tweets)
    update()

    # Save sarcasm data
    update("Saving sarcasm data...")
    count = len(sar_tweets)
    print("len pos_tweets before take = {}".format(len(pos_tweets)))
    non_sar_tweets = take(pos_tweets, count // 2) + take(neg_tweets, count // 2)
    print("len pos_tweets after take = {}".format(len(pos_tweets)))
    sar_labels = [1 for _ in sar_tweets]
    non_sar_labels = [0 for _ in non_sar_tweets]

    sarcasm_data = np.append(sar_tweets, non_sar_tweets)
    sarcasm_labels = np.append(sar_labels, non_sar_labels)

    sarcasm_data, sarcasm_labels = shuffle(sarcasm_data, sarcasm_labels)

    size = len(sarcasm_data)
    train = slice(0, int(0.8 * size))
    dev = slice(int(0.8 * size), int(0.9 * size))
    test = slice(int(0.8 * size), size - 1)

    sarcasm_dump = {"train" : (sarcasm_data[train], sarcasm_labels[train]),
                    "dev" : (sarcasm_data[dev], sarcasm_labels[dev]),
                    "test" : (sarcasm_data[test], sarcasm_labels[test])}

    pickle.dump(sarcasm_dump, open(os.path.join(SPLIT_DATA_DIR, "sarcasm.pkl"), 'wb'))
    update()

    # Save sentiment data
    update("Saving sentiment data...")
    count = min(len(pos_tweets), len(neg_tweets))
    pos_tweets = pos_tweets[:count]
    neg_tweets = neg_tweets[:count]
    pos_labels = [1 for _ in pos_tweets]
    neg_labels = [0 for _ in neg_tweets]

    sentiment_data = np.append(pos_tweets, neg_tweets)
    sentiment_labels = np.append(pos_labels, neg_labels)

    sentiment_data, sentiment_labels = shuffle(sentiment_data, sentiment_labels)

    size = len(sentiment_data)
    train = slice(0, int(0.8 * size))
    dev = slice(int(0.8 * size), int(0.9 * size))
    test = slice(int(0.8 * size), size - 1)

    sentiment_dump = {"train" : (sentiment_data[train], sentiment_labels[train]),
                      "dev" : (sentiment_data[dev], sentiment_labels[dev]),
                      "test" : (sentiment_data[test], sentiment_labels[test])}

    pickle.dump(sentiment_dump, open(os.path.join(SPLIT_DATA_DIR, "sentiment.pkl"), 'wb'))
    update()
    
def take(l, n):
    taken = l[:n]
    del l[:n]
    return taken

def update(message="DONE.\n"):
    print(message, end='', flush=True)

def tokenizer(text):
    return text.split(' ')


if __name__ == '__main__':
    main(POS_DIR, NEG_DIR, SAR_DIR, RAND_SEED)
