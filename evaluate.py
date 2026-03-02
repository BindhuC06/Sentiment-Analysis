from naive_bayes import predict_tweet

def test_naive_bayes(x_test, y_test, logprior, loglikelihood):
    accuracy = 0
    for tweet, y in zip(x_test, y_test):
        score = predict_tweet(tweet, logprior, loglikelihood)
        if score>0:
            prediction = 1
        else:
            prediction=0
        if prediction==y:
            accuracy+=1

    return accuracy / len(x_test)