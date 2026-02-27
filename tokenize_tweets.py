import nltk
from nltk.tokenize import TweetTokenizer

def tokenize(tweet):
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=True,
        reduce_len=True
    )
    tokens = tokenizer.tokenize(tweet)
    return tokens