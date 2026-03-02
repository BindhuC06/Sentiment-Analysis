import pandas as pd
import numpy as np
import nltk
from nltk.corpus import twitter_samples 
from clean_re import clean
from tokenize_tweets import tokenize
from stop_words_remove import stopclean
from stem_tweet import stems
from frequency import frequencies
from naive_bayes import train_naive_bayes, predict_tweet
from evaluate import test_naive_bayes

try:
    twitter_samples.fileids()
except LookupError:
    nltk.download('twitter_samples')

positive_data=twitter_samples.strings('positive_tweets.json')
negative_data=twitter_samples.strings('negative_tweets.json')

# visualizing 

from visualizing_raw_data import visual

# visual(positive_data,negative_data)

# cleaning the tweets using re library and 
# tokentizing the tweets -- splitting the tweets into individual words wint no tabs.

clean_positive=[]
clean_negative=[]

for tweet in positive_data:
    clean_positive.append(tokenize(clean(tweet)))
for tweet in negative_data:
    clean_negative.append(tokenize(clean(tweet)))

# removing stop words and punctuations

positive_clean_tokens=[stopclean(t) for t in clean_positive]
negative_clean_tokens=[stopclean(t) for t in clean_negative]

# stemming 

clean_tweet_positive=[stems(t) for t in positive_clean_tokens]
clean_tweet_negative=[stems(t) for t in negative_clean_tokens]

# ypos=np.ones(len(clean_tweet_positive))
# yneg=np.zeros(len(clean_tweet_negative))

# y=np.concatenate((ypos,yneg))

# total_clean_tweets=clean_tweet_negative+clean_tweet_positive

# splitting the data into training  and testing datasets.

all_positive = clean_tweet_positive
all_negative = clean_tweet_negative
#80:20
split_pos = int(0.8 * len(all_positive))
split_neg = int(0.8 * len(all_negative))

train_pos = all_positive[:split_pos]
test_pos  = all_positive[split_pos:]

train_neg = all_negative[:split_neg]
test_neg  = all_negative[split_neg:]

train_x=train_pos + train_neg
train_y=np.concatenate((np.ones(len(train_pos)),np.zeros(len(train_neg))))

x_test=test_pos + test_neg
y_test=np.concatenate((np.ones(len(test_pos)),np.zeros(len(test_neg)))) 

freq=frequencies(train_x,train_y)
logprior,loglikelihood=train_naive_bayes(freq,train_x,train_y)

accuracy=test_naive_bayes(x_test,y_test,logprior,loglikelihood)
print(f"Accuracy of the model is {accuracy*100:.2f} %")

while(True):
    tweet=input("Enter a tweet to predict its sentiment:")
    clean_tweet=stems(stopclean(tokenize(clean(tweet))))
    score=predict_tweet(clean_tweet,logprior,loglikelihood)
    if score>0:
        print("The tweet is positive")
    else:
        print("The tweet is negative")