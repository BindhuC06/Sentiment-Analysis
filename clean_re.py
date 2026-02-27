import re 
def clean(tweet):
    # remove oldschool tweets
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks in the tweet 
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtag symbol from the tweet
    tweet = re.sub(r'#', '', tweet)
    # removing mentions and @
    tweet = re.sub(r'@\w+', '', tweet)
    return tweet