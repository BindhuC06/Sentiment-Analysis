import numpy as np

def frequencies(tweets,ylab):
    """Building the frequencies.
    Input:
        tweets: a list of tweets
        ylab: an m x 1 array/matrix with the sentiment label of each tweet i.e 1 or 0
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    y=np.squeeze(ylab).tolist()
    freqs={}
    for i,tweet in zip(y,tweets):
        for word in tweet:
            pair=(word,i)
            # if pair in freqs:
            #     freqs[pair] += 1
            # else:
            #     freqs[pair] = 1
            freqs[pair] = freqs.get(pair, 0) + 1

    return freqs
