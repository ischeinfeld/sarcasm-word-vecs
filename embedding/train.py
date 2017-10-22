#!/usr/bin/env python

# import modules & set up logging
import gensim
import logging
from tweets import Tweets

# enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
# set up tweet iterator
tweet_dirs = ['../data/labeled_data/positive/',
              '../data/labeled_data/negative/']
tweets = Tweets(tweet_dirs) # iterator that returns preprocessed tweets

# train word2vec on the tweets
model = gensim.models.Word2Vec(tweets, iter=10, min_count=5, size=100)

# save word2vec model
model.save('./word2vec_model')
