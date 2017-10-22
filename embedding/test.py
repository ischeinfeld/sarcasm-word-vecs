#!/usr/bin/env python

# import modules & set up logging
import gensim
import logging

# enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
# set up tweet iterator
# load word2vec model
model = gensim.models.Word2Vec.load('./word2vec_model')

print("Example: happy is to sad as love is to: ", end='')
print(model.most_similar(positive=['happy', 'sad'], negative=['love'], topn=3))

print("Example: what doesn't match in \"happy, sad, glad, excited\": ", end='')
print(model.doesnt_match("happy sad glad excited".split()))

# trim unneeded model memory
embedding = model.wv
del model

# example vector
word = "kanye"
print("The embedding vector for \"{}\" is : ".format(word), end='')
print(embedding[word])
