import pickle
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
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
        
        # check if the tag is already in the 'classes' list
        if intent['tag'] not in classes:
            # if not then append it to the 'classes list
            classes.append(intent['tag'])
            
# apply lemmatization to each word and convert it to its base form, including only those words that are not in ignore_word list.
words = [lemmatizer.lemmatize(x.lower()) for x in words if x not in ignore_word]

# convert the list of words to a set inorder to remove any duplication, then convert them back to list, finally sorting the converted list in alphabetical order
words = sorted(list(set(words)))

# we follow the same convertion and sorting for classes as well, again to avoid any duplicate entries
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes")
print(len(words), "unique lemmatized words", words)

pickle.dump(words.open('wrd.pkl', 'wb'))
pickle.dump(classes.open('cls.pkl', 'wb'))



# create an empty list for training data
training = []

# create a list of zeroes with length equal to the number of unique elements in the class.
# this will serve as a template  for the output vectors, where each element represents a class.
output_empty = [0]*len(classes)

# now we loop through each document in the documents list
for doc in documents:
    # lets now initialize an empty list to hld the bag of words for each current document
    bag = []
    # Get the tokenized words for the given pattern from the current document
    pattern_word = doc[0]

    pattern_word = [lemmatizer.lemmatize(word.lower()) for word in pattern_word]
    
    for x in words:
        if x in pattern_word:
            bag.append(1)
        else:
            bag.append(0)
            
    
    # now, for each pattern, output is 0 for each tag and 1 for current tag
    output_row = list(output_empty)
    
    # set the element corresponding to the current document's class = 1
    output_row[classes.index(doc[1])] = 1
    
    # add the current bag of vector and its corresponding output to training list
    training.append([bag, output_row])
    
# now to randomize the order of input and output pairs, we will shuffle the training data
random.shuffle(training)

# for efficient numerical operations, we will now convert the shuffled training list into a numpy array
training = np.array(training)

# lets now extract the feature vector or patterns from the training data
train_x = list(training[:,0])

# and extract the labels or intents from the training data
train_y = list(training[:,1])

print("Training data successfully initiated.")

#---------------------------------------------------------------------------------------------------------------------------
# now we will be defining the model for which we will be using the sequential model from keras
model = Sequential()

# adding a layer of 128 neurons, RelU activation function, and input shape equal to the length of the feature veectors
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))

# to prevent overfitting we add a dropout layer
model.add(Dropout(0.5)) # dropout rate = 50%

# adding another layer of 64 neurons and ReLU activation function
model.add(Dense(64, activation='relu'))

# again add a droput layer of rate 50%
model.add(Dropout(0.5))

# now we finalize the output layer, with the number of neurons equal to number of classes
model.add(Dense(len(train_y[0]), activation='softmax'))

#---------------------------------------------------------------------------------------------------------------------------
# We now will compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
#---------------------------------------------------------------------------------------------------------------------------
# we now train and save the model
fin = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', fin)
print("Model creation")
        



