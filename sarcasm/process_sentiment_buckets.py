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

OUTFNAME_FORMAT = "preprocess_{}.tsv"
LOG_FNAME = "process.log"
INFNAME_FORMAT = "preprocess_{}.tsv"
OUTFNAME_FORMAT = "process_{}.pkl"
EMBEDDING_FNAME = "../embedding/word2vec_model"
CLF_FNAME = "../sentiment/clf_pipe.pkl"
RAND_SEED = 178
CUTOFF = 5
MAX_L_GIVEN = 100
MAX_NGRAM = 5 # must be <= max ngram size in sentiment model (5 at the moment)
RESOLUTION = 8

def read_data_file(data_fname, max_l):
    tweets = []
    change_count = 0
    
    with open(data_fname, "r") as infile:
        for line in infile:
            line = line.strip()
            line = line.lower()
            label, _, text = line.partition('\t');
            words = text.split()
            
            if len(words) > max_l:
                words = words[:max_l]
                change_count += 1

            datum = Datum(y=int(label),
                          num_words=len(words),
                          text=words,
                          ngrams=tweet_to_ngrams(words)) # list of sentiment np.arrays
            tweets.append(datum)
    
    print("length more than {}: {}".format(max_l, change_count)) 
    
    return tweets

def train_test(vocab):
    
    train_fname = INFNAME_FORMAT.format("train")
    test_fname = INFNAME_FORMAT.format("test")
    output_train_fname = OUTFNAME_FORMAT.format("train")
    output_test_fname = OUTFNAME_FORMAT.format("test")

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
    W, word_idx_map = get_W(w2v)

    S, ngram_idx_map = get_S()

    np.random.shuffle(train_data) # needed?
    pickle.dump((train_data, max_l, W, word_idx_map, S, ngram_idx_map), open(output_train_fname, "wb"))
    print("dataset created!")

    
def get_W(word_vecs):
    """
    Get word matrix and word index map.
    W[i] is the vector for word indexed by i.
    """
    vocab_size = len(word_vecs)
    k = len(next(iter(word_vecs.values()))) # dimension of embedding
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+2, k), dtype='float32')            
    
    W[0] = np.zeros(k) # 1st word is all zeros (for padding)
    W[1] = np.random.normal(0,0.17,k) # 2nd word is unknown word
    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def get_S():
    """
    Get sentiment bucket matrix.
    ngram_idx_map[ngram] is the index i for ngram such that
    S[i] is the approximate sentiment vector for ngram indexed by i.
    For MAX_NGRAM = 3, RESOLUTION = 4, 
       S = [[0.5, 0.5, 0.5]
            [0.0, 0.0, 0.25]
            [0.0, 0.0, 0.5]
            ...
            [0.0, 0.25, 0.0]
            ...
            [0.75, 0.75, 0.75]]
    """
    # get set of ngrams in text
    lines = open(INFNAME_FORMAT.format("train")).readlines() \
            + open(INFNAME_FORMAT.format("test")).readlines()
    raw = [process_line(l) for l in lines ]
    ngrams_in_data = set()
    for words in raw:
        for ngram in tweet_to_ngrams(words):
            ngrams_in_data.add(ngram)

    # generate sentiment array, each line is an approximate sentiment array
    neutral = np.array([[0.5] * MAX_NGRAM])
    steps = np.arange(0.0, 1.0, 1.0 / RESOLUTION) # ex. [0.0, 0.1, 0.2, ..., 0.9]
    S = cartesian([steps] * MAX_NGRAM)
    S = np.concatenate((neutral,S), axis=0)

    # load sentiment features from model
    clf_pipe = pickle.load(open(CLF_FNAME, 'rb')) # model

    vect = clf_pipe.best_estimator_.named_steps['vect']
    clf  = clf_pipe.best_estimator_.named_steps['clf']

    features_to_sent = vect.vocabulary_ # map from model features to sentiment index
    # currently sentiment = count_pos / (count_pos + count_neg)
    sentiments = clf.feature_count_[1,:] / np.sum(clf.feature_count_, axis=0)

    # get map from ngrams to index i of sentiment bucket in S
    # for ngram 1 2 3 the vector is the last S[i] such that
    # each value is less than (sent 3, sent 2 3, sent 1 2 3) 
    ngram_idx_map = {}
    for ngram in ngrams_in_data:
        index = 1 + get_sent_index(ngram, sentiments, features_to_sent) # 0 index is neutral
        ngram_idx_map[ngram] = index

    return S, ngram_idx_map

def get_sent_index(ngram, sentiments, features_to_sent):
    index = 0 
    for n in range(MAX_NGRAM): # for 1, 2, .., max_ngram length ngrams
        sub_ngram = ' '.join(ngram.split(' ')[-1 - n:])
        if sub_ngram in features_to_sent:
            sent = sentiments[features_to_sent[sub_ngram]]
        else:
            sent = 0.5

        # if RES = 10, sent = .15, bucket = 1 for [0.0, 0.1, 0.2, ...]
        bucket = min(int(sent * RESOLUTION), RESOLUTION - 1)
        index += (RESOLUTION ** n) * bucket

    return index


def create_vocab():
    """Returns a dictionary of [word] = [position]"""
    
    cutoff = CUTOFF
    vocab = defaultdict(float)
    
    lines = open(INFNAME_FORMAT.format("train")).readlines() \
            + open(INFNAME_FORMAT.format("test")).readlines()
    raw = [process_line(l) for l in lines ]
    cntx = Counter( [ w for e in raw for w in e ] )
    lst = [ x for x, y in cntx.items() if y > cutoff ]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    
    return vocab


def tweet_to_ngrams(tweet):
    tweet = ["## OOB ##"] * (MAX_NGRAM - 1) + tweet
    return list(map(' '.join, zip(*(tweet[i:] for i in range(MAX_NGRAM)))))
 

def load_embedding(fname, vocab):
    """
    Loads word embedding vecs from gensim Word2Vec model
    """
    model = gensim.models.Word2Vec.load(fname)
    embedding = model.wv # keep only the embedding dictionary
    del model # frees up memory used to store Word2Vec model
    restricted_embedding = {k: embedding[k] for k in vocab.keys() if k in embedding}
    return restricted_embedding


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
    train_test(vocab) # True placeholder
