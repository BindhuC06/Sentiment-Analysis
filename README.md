# Twitter Sentiment Analysis using Naive Bayes (NLP)

## Overview

This project implements a complete **Natural Language Processing (NLP) pipeline** that classifies tweets as **Positive** or **Negative**.
The model is built using probability theory and the Naive Bayes algorithm instead of high-level machine learning libraries.

The goal of the project is to demonstrate a clear understanding of:

* text preprocessing
* feature engineering
* probabilistic modeling
* supervised learning
* evaluation of a classifier

The system is trained and evaluated on the **NLTK Twitter Samples dataset (10,000 tweets)**.

---

## Key Features

• Tweet cleaning using Re library
• URL and @mentions removal
• Tokenization using NLTK TweetTokenizer
• Stopword removal
• Porter stemming
• Frequency dictionary construction
• Laplace (Add-1) smoothing
• Log-likelihood & log-prior computation
• Naive Bayes classification
• Accuracy evaluation
• Interactive user prediction

---

## Project Pipeline

1. **Data Loading**

   * Load positive and negative tweets from `twitter_samples` in NLTK library.

2. **Preprocessing**

   * Remove retweets, links, hashtags, and mentions
   * Normalize text to lowercase
   * Tokenize tweets into words
   * Remove stopwords and punctuations
   * Apply stemming using porter stemming

3. **Feature Extraction**

   * Build a frequency dictionary mapping
     `(word, sentiment) → count`

4. **Model Training**

   * Applied the Laplace smoothing
   * Computing:
     * Log Prior
     * Log Likelihood

5. **Prediction**

   * Calculate sentiment score:
     score = logprior + Σ loglikelihood(word)

   * If score > 0 → Positive
   * If score < 0 → Negative

6. **Evaluation**
   * 80-20 train-test split
   * Accuracy measured on unseen tweets

---

## Model Performance

Accuracy achieved: **99.55%**
Despite being a classical probabilistic model, Naive Bayes performs well on short informal text such as tweets.

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/BindhuC06/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Create virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the program

```bash
python main.py
```

---

## Example Input:

```
I really loved this movie!
```

## Expected Output:

```
The tweet is positive
```

## Example Input:

```
This is the worst product ever
```

## Expected Output:

```
The tweet is negative
```

---

## Technologies Used

* Python
* NLTK
* NumPy
* Regular Expressions
* Probability & Statistics

---

## What I Learned

* Building a classifier without using prebuilt ML models
* Importance of preprocessing in NLP
* Handling numerical underflow using logarithms
* Laplace smoothing for unseen words
* Model evaluation using train-test split

---

## Future Improvements

* Web interface using Streamlit
* Support for neutral sentiment
* Use of TF-IDF features
* Comparison with Logistic Regression and Deep Learning models

---

## Author

Bindhu
B.Tech Computer Science Student
Gitam University Hyderabad
GitHub: https://github.com/BindhuC06