"""
Extract situations from POS tagged tweets
"""

import os

class Tweets(object):

    def __init__(self, tweet_dirs, max_count = -1): # tweet_dir string of directory name
        self.tweet_dirs = tweet_dirs
        self.max_count = max_count

        self.filter_pos = set(['U', '~']) # URL or email address

    def __iter__(self):
        tweet_ids = set() # Set of tweet ids to check uniqueness
        
        count = 0
        for tweet_dir in self.tweet_dirs:
            for file in [f for f in os.listdir(tweet_dir) if f.endswith('.tsv')]:
                for line in open(os.path.join(tweet_dir, file)):
                    id = int(line.split('\t')[3])
                    if id not in tweet_ids:
                        tweet_ids.add(id)
                        yield self.preprocess(line)
                        count = count + 1
                        if self.max_count > 0 and count > self.max_count:
                            return

    def preprocess(self, line):

        columns = line.split('\t') # first three columns must be tokens, pos, confidences

        # [["token", ...], ["pos", ...], ["confidence", ...]]
        tweet = [columns[0].split(), columns[1].split(), columns[2].split()] 

        # [["token", ...], ["pos", ...], ["confidence", ...], [True, ...]]
        tweet = tweet + [[True] * len(tweet[0])] 

        # transpose to [["token", "pos", "confidence", True], ...]
        tweet = [list(x) for x in zip(*tweet)]   

        self.__filter(tweet)
       
        return [token[0] for token in tweet if token[3] == True]

    def __filter(self, tweet):
         
        for token in tweet:
            if token[1] in self.filter_pos:
                token[3] = False
    
    TAGS = ["#sarcasm", "#sarcastic", "#excited", "#grateful", "#happy",
            "#joy", "#loved", "#love", "#lucky", "#wonderful", "#angry",
            "#awful", "#disappointed", "#fear", "#frustrated", "#hate",
            "#sad", "#scared", "#stressed"]

    @classmethod
    def filter_tags(cls, tweet):
        """takes tweet as token list and removes gold label tags"""
        return [token for token in tweet if token.lower() not in cls.TAGS]
