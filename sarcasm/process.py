#!/usr/bin/env python

from collections import namedtuple
import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
from pprint import pprint
import gensim
import logging
from collections import Counter
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.utils.extmath import cartesian

from conv_net_classes import Datum

"""
assume input file: 1st column label (int), and 2nd column is tweet (string) with space separated tokens
"""

LOG_FNAME = "process.log"
INFNAME_FORMAT = "preprocess_{}.tsv"
OUTFNAME_FORMAT = "process_{}_{}.pkl"
EMBEDDING_FNAME = "../embedding/word2vec_model"
CLF_FNAME = "../sentiment/clf_pipe.pkl"
RAND_SEED = 178
CUTOFF = 0 #TODO check other values, and that they work with ngrams code
MAX_L_GIVEN = 100 # never reached in tweet data so far
MAX_NGRAM = 4 # must be <= max ngram size in sentiment model (5 at the moment), 0 for just embeddings

def read_data_file(data_fname, max_l):
    tweets = []
    change_count = 0
    
    with open(data_fname, "r") as infile:
        for line in infile:
#            line = line.strip()
#            line = line.lower()
            label, _, text = line.partition('\t');
            words = text.split()
            
            if len(words) > max_l:
                words = words[:max_l]
                change_count += 1

            datum = Datum(y=int(label),
                          num_words=len(words),
                          ngrams=tweet_to_ngrams(words)) 
            tweets.append(datum)
    
    print("length more than {}: {}".format(max_l, change_count)) 
    
    return tweets

def train_test(vocab):
    
    train_fname = INFNAME_FORMAT.format("train")
    test_fname = INFNAME_FORMAT.format("test")
    output_train_fname = OUTFNAME_FORMAT.format("train", MAX_NGRAM)
    output_test_fname = OUTFNAME_FORMAT.format("test", MAX_NGRAM)

    max_l = MAX_L_GIVEN
    
    print("loading data...")
    train_data = read_data_file(train_fname, max_l)
    test_data = read_data_file(test_fname, max_l)
    pickle.dump(test_data, open(output_test_fname, "wb"))

    train_lines = open(INFNAME_FORMAT.format("train")).readlines()
    max_l = max(map(lambda line: len(process_line(line)), train_lines))
    print("max_l is {}".format(max_l))
    
    print("data loaded!")
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    
    w2v_fname = EMBEDDING_FNAME
    print("loading word2vec vectors...")
    w2v = load_embedding(w2v_fname, vocab)
    
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))

    print("building WS, ngram_idx_map")
    WS, ngram_idx_map = get_WS(w2v)
    print("WS shape: {}".format(WS.shape))

    np.random.shuffle(train_data) # needed? #RANDOM

    pickle.dump((train_data, max_l, WS, ngram_idx_map), open(output_train_fname, "wb"))
    print("dataset created!")


def get_WS(w2v):
    """
    ngram_idx_map[ngram] is the index i for ngram such that
    WS[i] is the approximate embedding vector concatenating the embedding
    of the final word in the ngram and the sentiments of the subgrams
    """
    # get set of MAX_NGRAM-grams in text
    lines = open(INFNAME_FORMAT.format("train")).readlines() \
            + open(INFNAME_FORMAT.format("test")).readlines()
    raw = [process_line(l) for l in lines ]
    ngrams_in_data = set()
    for words in raw:
        for ngram in tweet_to_ngrams(words):
            ngrams_in_data.add(ngram)

    # load sentiment features from model
    clf_pipe = pickle.load(open(CLF_FNAME, 'rb')) # model

    vect = clf_pipe.best_estimator_.named_steps['vect']
    clf  = clf_pipe.best_estimator_.named_steps['clf']

    features_to_sent_idx = vect.vocabulary_ # map from model features to sentiment index
    # currently, sentiment = 2 * (count_pos / (count_pos + count_neg)) - 1
    sentiments = clf.feature_count_[1,:] / np.sum(clf.feature_count_, axis=0) # in [0,1]
    sentiments = 2 * sentiments - 1 # rescale to [-1,1]

    features_to_sent = {feat: sentiments[idx] for (feat,idx) in features_to_sent_idx.items()}

    # build WS and ngram_idx_map for each MAX_NGRAM-gram in the text
    k = len(next(iter(w2v.values()))) # dimension of embedding
    WS = np.zeros(shape=(len(ngrams_in_data) + 1, k + MAX_NGRAM), dtype='float32')
    ngram_idx_map = {}

    index = 1 # first row is left 0, for padding in the cnn. This is also neutral sentiment.
    # For Vader Sentiment analysis
#    vader_analyzer = SentimentIntensityAnalyzer()


    for ngram in ngrams_in_data:
        ngram_idx_map[ngram] = index

        # set word embedding, note that unknown words already randomized in load_embedding 
        words = ngram.split(' ')
        WS[index,:k] = w2v[words[-1]] # embedding of last word

        # set sentiment embedding
        for n in range(MAX_NGRAM): # for 1, 2, ... length ngrams
            sub_ngram = ' '.join(words[-1 - n:]) 

            # Naive Bayes Sentiment feature --------------------------------
            sent = features_to_sent.get(sub_ngram, 0.0) # default to neutral 0
            # --------------------------------------------------------------

#            # TextBlob sentiment feature -----------------------------------
#            sent = TextBlob(sub_ngram).sentiment.polarity
#            # --------------------------------------------------------------

#            # Vader sentiment feature -------------------------------------
#            sent = vader_analyzer.polarity_scores(sub_ngram)['compound']
#            # -------------------------------------------------------------
            WS[index,k+n] = sent

        index += 1

    return WS, ngram_idx_map

def tweet_to_ngrams(tweet):
    n = max(MAX_NGRAM, 1) # make ngrams of length MAX_NGRAM, except length 1 if it is 0
    tweet = ["## OOB ##"] * (n - 1) + tweet
    return list(map(' '.join, zip(*(tweet[i:] for i in range(n)))))
 

def create_vocab():
    """Returns a set of words in text"""
    
    cutoff = CUTOFF
    
    lines = open(INFNAME_FORMAT.format("train")).readlines() \
            + open(INFNAME_FORMAT.format("test")).readlines()
    raw = [process_line(l) for l in lines]
    cntx = Counter( [ w for e in raw for w in e ] )
    vocab = { x for x, y in cntx.items() if y > cutoff }
    
    return vocab


def load_embedding(fname, vocab):
    """
    Loads word embedding vecs from gensim Word2Vec model
    """
    model = gensim.models.Word2Vec.load(fname)
    embedding = model.wv # keep only the embedding dictionary
    del model # frees up memory used to store Word2Vec model

    k = len(embedding['a']) # dimension of embedding
    unknown_vec = lambda: np.random.normal(0,0.17,k) #TODO check these parameters
    
    restricted_embedding = {word: default_get(embedding, word, unknown_vec()) for word in vocab}
    return restricted_embedding


def default_get(embedding, word, default):
    try:
        return embedding[word]
    except:
        return default


def process_line(line):
    """Returns list of tokens in tweet"""
    [label, text] = line.split('\t')
    return text.split()

def tokenizer(text): 
    return text.split(' ')

if __name__=="__main__":
    # enable logging, #TODO add logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filename=LOG_FNAME, level=logging.INFO)

    # set random seed
    np.random.seed(RAND_SEED)

    # build vocab and sentiment
    print('create_vocab')
    vocab = create_vocab()

    print("vocab size is {}.".format(len(vocab)))
    print('train_test')
    train_test(vocab)
