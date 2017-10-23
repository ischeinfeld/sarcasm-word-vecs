# Data
Data preprocessing is handled in `/data`.

## Raw Data
Raw data is contained in `/data/raw_data/`. This is a directory of tab-seperated-variable files with the format

| tweet id | date / time | user | tweet text|

This is gathered into files `tweet.keyword` where each tweet in a file contains #keyword.

## Labeled Data
POS labeled tweets are contained in `/data/labeled_data/`. The [CMU tweet POS tagger](http://www.cs.cmu.edu/~ark/TweetNLP/) (contained in `/data/ark-tweet-nlp-0.3.2/` was used to tokenize and tag each tweet using the command `./runTagger.sh —input-field 2 —output-format pretsv` on a source file in `raw_data/`. The result for a source file was written to `labeled_data/sarcasm/`, `labeled_data/positive/`, or `/labeled_data/negative` depending on the keyword. Source file `tweet.keyword` was written to `keyword.tsv`. This has format

| tokenized tweet text | POS tags | confidences | tweet id | data / time | user | tweet text |


## Prepare Data for Processing
`/data/split_data.py` loads and filters the tokenized and POS labeled data with the `tweets` module contained in `/utils/`, removing the keyword hashtags and tokens tagged as URLs or special twitter identifiers. Two datasets are stored, `sarcasm.pkl` and `sentiment.pkl`. Each contains train, dev, and test sets of equal amounts of sarcastic/not sarcastic and positive/negative tweets, where the tweets have randomly been selected across all the representative keywords and shuffled.

# Embedding
Word embeddings are handled in `/embedding/`. Current embeddings are built with the default gensim model and are 100 dimensional.

**TODO** Add detail

# Sentiment
Training the sentiment model is handled in `/sentiment/`. A sentiment classifier is trained on ~1 million~ 800k each of positive and negative tweets (with keyword hashtags removed) using a Naive Bayes Classifier. This model classifies positive and negative tweets with an f1-score of 0.88. The features of this model will be used to augment word-vectors in sarcasm detection.

**TODO** Add detail

# Sarcasm
Sarcasm detection is handled in `/sarcasm/`. Here two CNN's can be trained, one to predict sarcasm from just word embeddings and one to predict sarcasm from word embedding vectors agumented with local sentiment features.

## Preprocessing
`preprocess.py` 

**TODO** Add detail

## Processing
`process.py`

**TODO** Add detail

## CNN

With 10 epochs, training on 19000 examples and testing on 1000.

| Ngram Sentiment Feature | Test Performance |
| ----------------------- |------------------| 
| 0 | 78.4 |
| 1 | 78.2 |
| 2 | 79.3 |
| 3 | 80.1 |
| 4 | 80.4 |
| 5 | 80.4 |
