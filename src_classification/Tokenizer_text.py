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

vocabulary_size=8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token="SENTENCE_START_TOKEN"

txt_flu,labels_flu=handle_flu_json("../data/flu.json.gz")
sentences=[]
for i in range(len(txt_flu)):
	sentences.append("{} {} {}".format(sentence_start_token,txt_flu[i],labels_flu[i]))

sentences=np.array(sentences)
for ele in sentences:
	pprint(ele)
	time.sleep(1)