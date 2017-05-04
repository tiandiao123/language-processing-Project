#!/usr/bin/env python
import numpy as np
import scipy as sp
import json
from pprint import pprint
import time
import math
import gzip
from process_data import handle_flu_json,handle_flu_risk_perception,handle_flu_vaccine_new
from process_data import handle_health_json,handle_trust_in_gov,handle_vaccine_sentiment,handle_zika_conspiracy
import nltk
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

vocabulary_size=8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token="SENTENCE_START_TOKEN"



txt_flu,labels_flu=handle_flu_json("../data/flu.json.gz")

def tokenize_words(txt,labels):
	sentences=[]
	for i in range(len(txt)):
		sentences.append("{} {} {}".format(sentence_start_token,txt[i],labels[i]))
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(sentences)
	sequences = tokenizer.texts_to_sequences(sentences)
	label_index=tokenizer.texts_to_sequences(labels)
	return sequences,label_index



# sequences,label_index=token_words(txt_flu,labels_flu)

# for i in range(len(sequences)):
# 	pprint(sequences[i])
# 	time.sleep(2)
# 	pprint(label_index[i])
# 	time.sleep(1)
