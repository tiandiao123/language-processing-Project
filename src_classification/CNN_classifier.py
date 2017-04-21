#!/usr/bin/env python

#The following codes are used to handle flu.json and detect whether the flu-relavant factor is related to the comments of twitter

import numpy as np
import scipy as sp
import json
from pprint import pprint
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


# this function is used to extract data from flu.json file
def handle_flu_json(path):
	with open(path) as data_file:
		json_data = [json.loads(line) for line in data_file]
	train_txt=[]
	train_label=[]
	label=[]
	for i in range(len(json_data)):
		train_txt.append(json_data[i]['text'])
		if 'flu_relevant' in json_data[i]['label']:
			train_label.append(json_data[i]['label']['flu_relevant'])
		else:
			train_label.append('false')
	return train_txt,train_label

# load the dataset
txt,labels=handle_flu_json("../data/flu.json")

#default value for hadnling large scale data
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000

tokenizer = Tokenizer()
tokenizer.fit_on_texts(txt)
sequences = tokenizer.texts_to_sequences(txt)


#convert labels into binary label
prediction_label=[]
for i in range(len(labels)):
    if labels[i]=='false':
        prediction_label.append(0)
    elif labels[i]=='true':
        prediction_label.append(1)

data = pad_sequences(sequences, maxlen=40)

#split the data into training data and test data
X_train,X_test,y_train,y_test=train_test_split(data,prediction_label,test_size=0.33)
word_index = tokenizer.word_index
EMBEDDING_DIM = 80
num_words = len(word_index)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))


#CNN model
input_sentence = Input(shape=(40,), dtype='int32')
embedded_sequences = Embedding(num_words+1, EMBEDDING_DIM)(input_sentence)
x = Conv1D(80, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(3)(x)
x = Conv1D(60, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Flatten()(x)
x = Dense(60, activation='relu')(x)
x=Dropout(0.3)(x)
x=Dense(30,activation='relu')(x)
x=Dropout(0.35)(x)
x=Dense(10,activation='relu')(x)
preds=Dense(1,activation='relu')(x)

model = Model(input_sentence, preds)
model.compile(loss='mse',
              optimizer='sgd',
              metrics=['acc'])



#training our model!
model.fit(X_train, y_train, validation_data=(X_test, y_test),nb_epoch=10, batch_size=128)
