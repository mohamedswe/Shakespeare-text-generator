# importing the required packages 

import random 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

#Getting the Shakespeare text 

filepath= tf.keras.utils.get_file('shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text= open(filepath,'rb').read().decode(encoding='utf-8').lower()

#Preparing the data 

#training the data on 500k characters 
text= text[200000:700000]

# Getting all the unique characters 
 
characters = sorted(set(text))

#Changing the unique character from char form to int form 

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

#Constant variables that dictate how long a "sentence should be"
#We also define how many chars we skip before starting the new sentence 

SEQ_LENGTH = 40
STEP_SIZE=3

#We are defining the training data and the target data 
sentences = []
next_char = []

# Here we are filling sentences and next_char 
# We iterate through the entire text and gather sentences and their next char 
for i in range(0,len(text)-SEQ_LENGTH,STEP_SIZE):
  sentences.append(text[i : i + SEQ_LENGTH])
  next_char.append(text[i + SEQ_LENGTH])

#The training data we gathered is in char format we need to change it to numerical 
#and save it into a numpy array 

x = np.zeros((len(sentences), SEQ_LENGTH,
              len(characters)), dtype=np.bool)
y = np.zeros((len(sentences),
              len(characters)), dtype=np.bool)
for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1

# Now that we have all the data it is time to create the model 

model = Sequential()
model.add(LSTM(128,
               input_shape=(SEQ_LENGTH,
                            len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01))

model.fit(x, y, batch_size=256, epochs=4)

#This is a helper function that was copied from the keras tutorial in order to generate reasonable text

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#This is the function used to generate the text it takes in 2 input parameters the length and temperature 
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                                 temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated


print(generate_text(300, 0.5))


