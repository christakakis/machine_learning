'''
Panagiotis Christakakis
AIDA, Dept. of Apllied Informatics, UoM
aid23004@uom.edu.gr
ID: aid23004
'''

#Import libraries
import nltk
from nltk.util import bigrams, trigrams
import random
from urllib import request

# Download the gutenberg and punkt package
nltk.download('punkt')
nltk.download('gutenberg')

# Set the list of fileIDs for the books you want to download
book_file_ids = ['melville-moby_dick.txt', 'whitman-leaves.txt', 'melville-moby_dick.txt', 'carroll-alice.txt', 'austen-emma.txt',
                'bryant-stories.txt', 'chesterton-brown.txt', 'shakespeare-hamlet.txt', 'austen-sense.txt', 'edgeworth-parents.txt']

# Initialize the combined text string
combined_text = ''

# Loop through the list of fileIDs
for file_id in book_file_ids:
    # Get the raw text of the current book
    book_text = nltk.corpus.gutenberg.raw(file_id)
    # Append the book text to the combined text string
    combined_text += book_text

# Open a new file for writing the combined text
with open('combined_books.txt', 'w') as combined_file:
    # Write the combined text to the new file
    combined_file.write(combined_text)

# Set the text file you want to be tokenized
text_file = open("combined_books.txt").read()

# Split the text into sentences
sentences = nltk.sent_tokenize(text_file)

# Initialize an empty list to hold the tokens
tokens = []

# Loop through the sentences
for sentence in sentences:
    # Tokenize the sentence and add the tokens to the list
    tokens += nltk.word_tokenize(sentence)

# Generate a list of unigrams, bigrams and trigrams from the tokens
unigrams = list(nltk.ngrams(tokens, 1))
bigrams = list(nltk.ngrams(tokens, 2))
trigrams = list(nltk.ngrams(tokens, 3))

# Count the frequency of each unigram, bigram and trigram
unigram_freq = nltk.FreqDist(unigrams)
bigram_freq = nltk.FreqDist(bigrams)
trigram_freq = nltk.FreqDist(trigrams)

# Create two new lists for every turple of words of calculated bigram frequencies
first_tuple_bigrams = [x[0] for x in bigram_freq]
second_tuple_bigrams = [x[1] for x in bigram_freq]

# Create three new lists for every turple of words of calculated bigram frequencies
first_tuple_trigrams = [x[0] for x in trigram_freq]
second_tuple_trigrams = [x[1] for x in trigram_freq]
third_tuple_trigrams = [x[2] for x in trigram_freq]

# The following functions generate bigram and trigram sentences
# by sampling based on the calculated frequencies found before.
# An upper limit of 100 words has been set if the end token is 
# not found yet, in order to get rid of the very long sentences.
# They are almost the same with some exceptions.

def generate_bigram_sentence(max_words):
  # Initialize an empty list to hold the generated bigrams.
  generated_bigrams = []
  # Randomly pick a starting token, which must only be a word 
  # from a-z & A-Z and append it to the sentence as the first one.
  next_word = " "
  while not next_word.isalpha():
    next_word = random.choice(first_tuple_bigrams)
  generated_bigrams.append(next_word)
  # Initialize end token.
  end_token = "."
  # While sentence is smaller than the max words.
  while len(generated_bigrams) < max_words:
    # Initialize an empty list every time to hold
    # all the second words of our next_word.
    sec_word = []
    # Iterate through the length of the list of tuples
    for i in range(len(first_tuple_bigrams)):
      # Every time the word we currently have is found,
      # its next word is appended to the list.
      if first_tuple_bigrams[i] == next_word:
        sec_word.append(second_tuple_bigrams[i])
    # With random sampling we choose the next word.
    next_word = random.choice(sec_word)
    generated_bigrams.append(next_word)
    # If the end token is found, we break the loop.
    if next_word == end_token:
      break

  # The final sentence is returned with a blank between every word
  return " ".join(generated_bigrams)

def generate_trigram_sentence(max_words):
  # Initialize an empty list to hold the generated trigrams
  generated_trigrams = []
  # Initialize the two starting tokens, the end token 
  # and append the starting ones to the sentence.
  start_word_one = "It"
  start_word_two = "is"
  end_token = "."
  generated_trigrams.append(start_word_one)
  generated_trigrams.append(start_word_two)
  # Iterate through the length of the list of tuples
  while len(generated_trigrams) < max_words:
    third_word = []
    # Just like the bigram sentence generator, this time an extra
    # for loop is necessary to find all the second and third words
    # that accompany the ones we have and then we random sampling
    # again takes place to find the next word.
    for i in range(len(first_tuple_trigrams)):
      if first_tuple_trigrams[i] == start_word_one:
        for j in range(len(second_tuple_trigrams)):
          if second_tuple_trigrams[j] == start_word_two:
            third_word.append(third_tuple_trigrams[j])
    next_word = random.choice(third_word)
    generated_trigrams.append(next_word)
    if next_word == end_token:
      break
  
  # The final sentence is returned with a blank between every word
  return " ".join(generated_trigrams)

# Please select the number of sentences you want to generate
# with bigrams and trigrams and the sentence maximum words.
num_of_sentences = 10
max_words = 30

print("\nCurrently Printing Sentences Generated By Bigrams:")
for i in range(num_of_sentences):
  print("\nSentence ", i, "\n", generate_bigram_sentence(max_words))

print("\nCurrently Printing Sentences Generated By Trigrams:")
for i in range(num_of_sentences):
  print("\nSentence ", i, "\n", generate_trigram_sentence(max_words))