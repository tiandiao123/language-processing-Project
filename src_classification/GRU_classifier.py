import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from pprint import pprint
from process_data import handle_flu_json,handle_flu_risk_perception,handle_flu_vaccine_new
from process_data import handle_health_json,handle_trust_in_gov,handle_vaccine_sentiment,handle_zika_conspiracy
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from Tokenizer_text import tokenize_words
import time
import Tokenizer_text


def GRU_train_prediction(sequences,labels,max_review_length):
	#split the data into training data and test data
	top_words = 10000
	labels=np.array(labels)
	label_array=labels.flatten()

	label2index=list(set(label_array))
	prediction_labels=[]
	
	for i in range(len(labels)):
		if labels[i]==label2index[0]:
			prediction_labels.append(0)
		else:
			prediction_labels.append(1)

	prediction_labels=np.array(prediction_labels)
	X_train,X_test,y_train,y_test=train_test_split(sequences,prediction_labels,test_size=0.33)
	#max_review_length = 500
	X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
	X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
	X_train=np.array(X_train)
	X_test=np.array(X_test)
	print(X_train.shape)
	print(X_test.shape)

	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
	model.add(GRU(50))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	

top_words = 6000
txt_flu,labels_flu=handle_flu_json("../data/flu.json.gz")
sequences,labels=tokenize_words(txt_flu,labels_flu)
GRU_train_prediction(sequences,labels,40)

#split the data into training data and test data
# top_words = 5000
# labels=np.array(labels)
# label_array=labels.flatten()

# label2index=list(set(label_array))
# prediction_labels=[]
# for i in range(len(labels)):
# 	if labels[i]==label2index[0]:
# 		prediction_labels.append(0)
# 	else:
# 		prediction_labels.append(1)

# prediction_labels=np.array(prediction_labels)
# X_train,X_test,y_train,y_test=train_test_split(sequences,prediction_labels,test_size=0.33)

# max_review_length = 40
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# X_train=np.array(X_train)
# X_test=np.array(X_test)


# for ele in X_train:
# 	print(ele)
# 	time.sleep(1)

# print(X_train.shape)





###################################################################################################

# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# max_review_length = 500
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# print(X_train.shape)

# for ele in X_train:
# 	print(type(ele))
# 	time.sleep(1)

# create the model
# embedding_vecor_length = 32
# model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# #model.add(Dropout(0.2))
# model.add(LSTM(100))
# #model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, epochs=3, batch_size=64)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))


