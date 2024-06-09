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
ignore_words = ['?', '!']
data_src = open('data.json').read()
intents = json.loads(data_src)

# Step 1: Extract list of intents from a dictionary called intents
for intent in intents['intents']:
    # Loop through all patterns in current intent
    for pattern in intent['patterns']:
        # Split the patterns, that is, the sentences or phrases, into words and assign them to a variable
        w = nltk.word_tokenize(pattern)
        words.extend(w)  # Add the tokens from the current pattern to the "words" list
        # Store tuples in a list that stores the tuples corresponding to each tag
        documents.append((w, intent['tag']))
        
        # Check if the tag is already in the 'classes' list
        if intent['tag'] not in classes:
            # If not then append it to the 'classes' list
            classes.append(intent['tag'])
            
# Apply lemmatization to each word and convert it to its base form, including only those words that are not in ignore_words list.
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# Convert the list of words to a set in order to remove any duplication, then convert them back to list, finally sorting the converted list in alphabetical order
words = sorted(list(set(words)))

# We follow the same conversion and sorting for classes as well, again to avoid any duplicate entries
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Create an empty list for training data
training = []

# Create a list of zeroes with length equal to the number of unique elements in the class.
# This will serve as a template for the output vectors, where each element represents a class.
output_empty = [0] * len(classes)

# Now we loop through each document in the documents list
for doc in documents:
    # Initialize an empty list to hold the bag of words for each current document
    bag = []
    # Get the tokenized words for the given pattern from the current document
    pattern_words = doc[0]

    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
            
    # Now, for each pattern, output is 0 for each tag and 1 for current tag
    output_row = list(output_empty)
    
    # Set the element corresponding to the current document's class = 1
    output_row[classes.index(doc[1])] = 1
    
    # Add the current bag of vector and its corresponding output to training list
    training.append([bag, output_row])
    
# Now to randomize the order of input and output pairs, we will shuffle the training data
random.shuffle(training)

# Extract the feature vector or patterns from the training data
train_x = np.array([np.array(item[0]) for item in training])

# Extract the labels or intents from the training data
train_y = np.array([np.array(item[1]) for item in training])

print("Training data successfully initiated.")

#---------------------------------------------------------------------------------------------------------------------------
# Now we will be defining the model for which we will be using the sequential model from keras
model = Sequential()

# Adding a layer of 128 neurons, ReLU activation function, and input shape equal to the length of the feature vectors
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))

# To prevent overfitting we add a dropout layer
model.add(Dropout(0.5))  # Dropout rate = 50%

# Adding another layer of 64 neurons and ReLU activation function
model.add(Dense(64, activation='relu'))

# Again add a dropout layer of rate 50%
model.add(Dropout(0.5))

# Now we finalize the output layer, with the number of neurons equal to number of classes
model.add(Dense(len(train_y[0]), activation='softmax'))

#---------------------------------------------------------------------------------------------------------------------------
# We now will compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
#---------------------------------------------------------------------------------------------------------------------------
# We now train and save the model
fin = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', fin)
print("Model creation")

