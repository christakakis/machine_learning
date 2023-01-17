'''
Panagiotis Christakakis
AIDA, Dept. of Apllied Informatics, UoM
aid23004@uom.edu.gr
ID: aid23004
'''

# Import libraries
import nltk
import string
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk import FreqDist
from random import shuffle
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

# Download additional packages
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')

# In order to have better results we have to extract the most
# valuable features of the reviews we have. For that we remove
# symbols and stopwords that don't really help us. 
stopwords_english = stopwords.words('english')

# This function cleans the words list and creates a new
# dictionary with the words we truly need.
def bag_of_words(words):
    words_clean = []
    for word in words:
        word = word.lower()
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)
    words_dictionary = dict([word, True] for word in words_clean)

    # Returning the dictionary
    return words_dictionary

# Each bag of words is assigned with a category
pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)

neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)

# Here each review gets a positive or negative category.
# positive reviews feature set
pos_reviews_set = []
for words in pos_reviews:
    pos_reviews_set.append((bag_of_words(words), 'pos'))

# negative reviews feature set
neg_reviews_set = []
for words in neg_reviews:
    neg_reviews_set.append((bag_of_words(words), 'neg'))

# Shuffling positive and negative reviews while also trying 
# to mantain our data stratified. This means our train and 
# test data have the same proportion of positive and negative
# reviews by the time they get split. So our classifier should
# have enough samples to correctly find the reviews outcome.
# Lastly we split our data into 80 - 20 ratio.

shuffle(pos_reviews_set)
shuffle(neg_reviews_set)

test_set = pos_reviews_set[:200] + neg_reviews_set[:200]
train_set = pos_reviews_set[200:] + neg_reviews_set[200:]

# Naive Bayes Classifier is used for training
classifier = NaiveBayesClassifier.train(train_set)

accuracy = classify.accuracy(classifier, test_set)
print("Accuracy of Naive Bayes is: ", accuracy)

custom_review = "This film was awful. Acting was poor and the general picture was a disaster."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_words(custom_review_tokens)
print (classifier.classify(custom_review_set))
# Negative review correctly classified as negative

# probability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result)
print (prob_result.max())
print (prob_result.prob("neg"))
print (prob_result.prob("pos"))

custom_review = "Loved the actors. The story was amazing and direction at it's best. Wonderful movie."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_words(custom_review_tokens)

print (classifier.classify(custom_review_set))
# Positive review correctly classified as positive

# probability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result)
print (prob_result.max())
print (prob_result.prob("neg")) 
print (prob_result.prob("pos"))