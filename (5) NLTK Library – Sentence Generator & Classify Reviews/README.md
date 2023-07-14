# NLTK Library – Sentence Generator & Classify Reviews
## Natural Language Processing with NLTK

This repository contains Python code for natural language processing tasks using the NLTK library. The goal is to demonstrate the usage of NLTK for text analysis and classification.

## Part A: Text Analysis using NLTK

In this section, we will focus on using NLTK to perform text analysis on 10 books from the [Gutenberg project](https://www.gutenberg.org/) corpus. The following steps are be included:

1. Load 10 books of your choice from the Gutenberg project corpus using NLTK.
2. Create a dictionary from all the words that appear in the books using a tokenizer of our choice (NLTK provides several options).
3. Divide the texts into sentences using a sentence tokenizer.
4. Calculate the frequencies of unigrams, bigrams, and trigrams. For unigrams and trigrams, we define a sentence start token and a sentence end token. For trigrams, we specify two sentence start tokens and one sentence end token.
5. Generate and report on our work using the diagrams (unigrams and bigrams) and the trigrams.

## Part B: Sentiment Analysis on Movie Reviews

In this section, we will focus on performing sentiment analysis on the [movie_reviews dataset](https://www.nltk.org/howto/corpus.html?highlight=movie_reviews) using NLTK. The dataset contains 2,000 movie reviews, with 1,000 positive (pos) reviews and 1,000 negative (neg) reviews.

The steps included are:

1. Load the movie_reviews dataset from NLTK, which contains the movie reviews.
2. Train a classifier (e.g., Naïve Bayes classifier, neural network, etc.) to correctly classify positive and negative reviews.
3. Choose the features to use as input for the classifier, such as individual words or n-grams.

---

This repository was initially created to store personal Python codes but is also available to others interested in similar projects. The codes contained in this repository were specifically developed for a Machine Learning and Natural Language Processing course in the MSc program, as part of a natural language processing and text classification project.

Please note that the data used in this repository belongs to their respective owners, and the repository's purpose is to showcase analytical skills and code implementation.
