#!/usr/bin/env python
import numpy as np
from pprint import pprint
import time
import math
from process_data import handle_flu_json,handle_flu_risk_perception,handle_flu_vaccine_new
from process_data import handle_health_json,handle_trust_in_gov,handle_vaccine_sentiment,handle_zika_conspiracy
import nltk
from keras.preprocessing.text import Tokenizer

vocabulary_size=8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token="SENTENCE_START_TOKEN"



def tokenize_words(txt,labels):
	sentences=[]
	for i in range(len(txt)):
		sentences.append("{} {} {}".format(sentence_start_token,txt[i],labels[i]))
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(sentences)
	sequences = tokenizer.texts_to_sequences(sentences)
	label_index=tokenizer.texts_to_sequences(labels)
	return sequences,label_index

