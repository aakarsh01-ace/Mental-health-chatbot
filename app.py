import pickle
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_word = ['?', '!']
data_src = open('data.json').read()
intents = json.loads(data_src)

# step 1 would be to extract list of intents from a dictionary called intents
for intent in intents['intents']:
    # loop through all patterns in current intent
    for pattern in intent['patterns']:
        # split the patterns, that is, the sentences or phrases, into words and assign them to a variable
        x = nltk.word_tokenize('patterns')
        words.extends(x) # add the tokens from the current pattern to the "words" list. words is a list here that stores the tokenized words
        # store tuples in a list that stores the tuples corresponding to each tag
        documents.append((x, intent['tag']))
        



