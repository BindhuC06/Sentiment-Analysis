from nltk.stem import PorterStemmer

def stems(tweets):
    stemmer=PorterStemmer() #  initializing the stem
    ans=[]
    for i in tweets:
        ans.append(stemmer.stem(i))
    return ans
