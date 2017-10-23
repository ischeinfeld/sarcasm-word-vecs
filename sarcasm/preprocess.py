#!/usr/bin/env python

import os
import sys
import argparse
import random
import numpy as np 
from pprint import pprint
import itertools
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn import metrics
import gensim
import logging

from tweets import Tweets

RAND_SEED = 1782
OUTFNAME_FORMAT = "preprocess_{}.tsv"
LOG_FNAME = "preprocess.log"

def main(arguments):

    # enable logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filename=LOG_FNAME, level=logging.INFO)

    # parse optional filename arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--sartic-tweets', dest='sar_dir',
                        help="Directory of example sartic tweets",
                        default="../data/labeled_data/sarcastic/")
    parser.add_argument('-p', '--positive-tweets', dest='pos_dir',
                        help="Directory of example positive tweets",
                        default="../data/labeled_data/positive/")
    parser.add_argument('-n', '--negative-tweets', dest='neg_dir',
                        help="Directory of example negative tweets",
                        default="../data/labeled_data/negative/")
    parser.add_argument('-c', '--sample-count', dest='sample_count',
                        help="Max number of samples of each class",
                        default="10000") # 10k default, ~300k max with current data

    args = parser.parse_args(arguments)

    # set random seed
    np.random.seed(RAND_SEED)

    # create tweets iterators
    log_print("Creating tweet iterators...")
    sar_tweets_iter = Tweets([args.sar_dir])
    pos_tweets_iter = Tweets([args.pos_dir])
    neg_tweets_iter = Tweets([args.neg_dir])
    log_print()

    # load tweets with gold labels filtered to lists and shuffle
    log_print("Loading sarcastic tweets with gold labels filtered...")
    sar_tweets = [Tweets.filter_tags(tweet) for tweet in sar_tweets_iter]
    log_print("...loaded {} sarcastic tweets".format(len(sar_tweets)))

    log_print("Loading non-sarcastic tweets...")
    pos_tweets = [Tweets.filter_tags(tweet) for tweet in pos_tweets_iter] # filter gold label hashtags
    log_print("...loaded {} positive tweets...".format(len(pos_tweets)))
    neg_tweets = [Tweets.filter_tags(tweet) for tweet in neg_tweets_iter]
    log_print("...loaded {} negative tweets".format(len(neg_tweets)))

    log_print("Selecting balanced sample sets of {} tweets per class...".format(args.sample_count))
    sample_count = int(args.sample_count)
    sar_tweets = resample(sar_tweets, n_samples=sample_count,
                              replace=False, random_state=1)
    pos_tweets = resample(pos_tweets, n_samples=sample_count//2,
                              replace=False, random_state=2)
    neg_tweets = resample(neg_tweets, n_samples=sample_count//2,
                              replace=False, random_state=3)
    non_tweets = pos_tweets + neg_tweets
    log_print()

    # shuffle tweets and split into training, dev, and test
    log_print("Shuffle all tweets...")
    sar_labels = [1 for _ in sar_tweets]
    non_labels = [0 for _ in non_tweets]

    tweets = np.append(sar_tweets, non_tweets)
    labels = np.append(sar_labels, non_labels)

    tweets, labels = shuffle(tweets, labels, random_state=4)
    log_print()
    
    # write to output file
    log_print("write to files as training, dev, and test sets...")
    output_gen = (n for n in zip(tweets, labels)) # generator of (tweet, label) tuples
    with open(OUTFNAME_FORMAT.format("test"), "w+") as f:
        for tweet, label in itertools.islice(output_gen, sample_count // 10):
            f.write("{}\t{}\n".format(label, ' '.join(tweet)))
    with open(OUTFNAME_FORMAT.format("dev"), "w+") as f:
        for tweet, label in itertools.islice(output_gen, sample_count // 10):
            f.write("{}\t{}\n".format(label, ' '.join(tweet)))
    with open(OUTFNAME_FORMAT.format("train"), "w+") as f:
        for tweet, label in output_gen:
            f.write("{}\t{}\n".format(label, ' '.join(tweet)))

    log_print("...training, dev, and test sets written to files {}, {}, and {}".
              format(OUTFNAME_FORMAT.format("train"), OUTFNAME_FORMAT.format("dev"), OUTFNAME_FORMAT.format("test")))
                

def log_print(message="DONE"):
    logging.info(message)
    if message[-3:] == "...":
        end = ""
    else:
        end = "\n"
    print(message, end=end, flush=True)

def tokenizer(text):
    return text.split(' ')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
