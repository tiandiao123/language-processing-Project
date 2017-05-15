import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import time
# fix random seed for reproducibility
np.random.seed(7)

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

for i in range(len(X_train)):
	print(X_train[i])
	time.sleep(1)
	print(y_train[i])