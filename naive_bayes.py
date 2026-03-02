import numpy as np

def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0
    # set of all the unique words in the training set
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    # word counts
    pos = 0
    neg = 0
    for pair in freqs:
        if pair[1] == 1:
            pos += freqs[pair]
        else:
            neg += freqs[pair]

    D = len(train_y)
    D_pos = sum(train_y)
    D_neg = D - D_pos
    #prior ratio =1 in case of equal number of positive and negative tweets. and hence log prior is 0.
    logprior = np.log(D_pos) - np.log(D_neg)
    for word in vocab:
        freq_pos = freqs.get((word,1),0)
        freq_neg = freqs.get((word,0),0)
        
        # Laplace smoothing for each class
        p_w_pos = (freq_pos + 1) / (pos + V)
        p_w_neg = (freq_neg + 1) / (neg + V)

        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood

def predict_tweet(tokens, logprior, loglikelihood):
    score = 0
    score += logprior
    for word in tokens:
        if word in loglikelihood:
            score += loglikelihood[word]
    return score