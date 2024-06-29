import json 
import numpy as np
from .nltk_utils import tokenize, stem, bag_of_words
import os

import tensorflow as tf

# Train

ignore_words = ["?", "i", "it", "a", "!", "me", "can", "you", "'s", "does", "of", "do", "my"]

intents_cur_dir = os.path.dirname(__file__) + "\intents.json"

with open(intents_cur_dir) as f:
    intents = json.load(f)

tags = []
all_words = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    
    for pattern in intent["patterns"]:
        
        words = tokenize(pattern)
        l1 = [stem(w) for w in words if w.lower() not in ignore_words]
        
        all_words.extend(l1)
        xy.append((l1, tag))

all_words = sorted(set(all_words))
np.random.shuffle(xy)

# Prepare training set

X_train = []
y_train = []

for sentence, tag in xy:
    bag = bag_of_words(sentence, all_words)
    X_train.append(bag)

    label = bag_of_words(tag, tags)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# # Model

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=16, input_shape = np.shape(all_words), activation='relu'))
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

ann.add(tf.keras.layers.Dense(units=len(tags) , activation='softmax'))

ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy']) 

ann.fit(X_train, y_train, batch_size=6, epochs = 200)

# # Response

def respond(sentence):

    '''
    input : 
        sentence : input prompt from user

    output : 
        response : text from given set of responses in training data
        tag :      tag of input prompt
    '''

    words = tokenize(sentence)
    sentence = [stem(w) for w in words if w not in ignore_words]
    bag = bag_of_words(sentence, all_words).reshape(1, -1)
    
    probs = ann.predict(bag)

    if max(probs[0]) < 0.4 :
        max_idx = 0
    else :
        max_idx = np.argmax(probs)
    tag = intents['intents'][max_idx]['tag']
    response = np.random.choice(intents['intents'][max_idx]['responses'])
    
    return response, tag
