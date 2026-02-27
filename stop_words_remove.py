import nltk
import string
from nltk.corpus import stopwords

def stopclean(clean_tokens):
    clean_tweets=[]

    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    for i in clean_tokens:
        i=i.strip()
        if (i not in stop_words and i not in string.punctuation and i!='.' and i!=''):
            clean_tweets.append(i)

    return clean_tweets