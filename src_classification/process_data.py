#!/usr/bin/env python
import numpy as np
import scipy as sp
import json
from pprint import pprint
import time
import math
import gzip

##extract data:txt, labels:flu-relavant
def handle_flu_json(path):
	json_data=[]
	with gzip.open(path, 'rb') as f:
		file_content = f.readlines()
		for line in file_content:
			json_data.append(json.loads(line.decode("utf-8")))

	train_txt=[]
	train_label=[]
	for i in range(len(json_data)):
		if 'flu_relevant' in json_data[i]['label']:
			train_label.append(json_data[i]['label']['flu_relevant'])
			train_txt.append(json_data[i]['text'])
	return train_txt,train_label

## extract data of train_data,label_about_flu,label_about_fluShot,label_about_flu_likelihood,label_about_flu_severity
def handle_flu_risk_perception(path):
	json_data=[]
	with gzip.open(path, 'rb') as f:
		file_content = f.readlines()
		for line in file_content:
			json_data.append(json.loads(line.decode("utf-8")))

	train_data=[]
	label_about_flu=[]
	label_about_fluShot=[]
	label_about_flu_likelihood=[]
	label_about_flu_severity=[]

	for lines in json_data:
		train_data.append(lines['text'])
		label_about_flu.append(lines['label']['about_flu'])
		label_about_fluShot.append(lines['label']['about_fluShot'])
		label_about_flu_likelihood.append(lines['label']['about_flu_likelihood'])
		label_about_flu_severity.append(lines['label']['about_flu_severity'])
	return train_data,label_about_flu,label_about_fluShot,label_about_flu_likelihood,label_about_flu_severity

# extract data of data,label_flu_vaccine_intent_to_receive,label_flu_vaccine_received,label_flu_vaccine_relevant
#,label_flu_vaccine_sentiment
def handle_flu_vaccine_new(path):
	json_data=[]
	with gzip.open(path, 'rb') as f:
		file_content = f.readlines()
		for line in file_content:
			json_data.append(json.loads(line.decode("utf-8")))

	data=[]
	label_flu_vaccine_intent_to_receive=[]
	label_flu_vaccine_received=[]
	label_flu_vaccine_relevant=[]
	label_flu_vaccine_sentiment=[]

	for line in json_data:
		data.append(line['tweet']['text'])

		if 'flu_vaccine_intent_to_receive' in line['label']:
			label_flu_vaccine_intent_to_receive.append(line['label']['flu_vaccine_intent_to_receive'])
		else:
			label_flu_vaccine_intent_to_receive.append(None)
		
		if 'flu_vaccine_received' in line['label']:
			label_flu_vaccine_received.append(line['label']['flu_vaccine_received'])
		else:
			label_flu_vaccine_received.append(None)

		if 'flu_vaccine_relevant' in line['label']:
			label_flu_vaccine_relevant.append(line['label']['flu_vaccine_relevant'])
		else:
			label_flu_vaccine_received.append(None)

		if 'flu_vaccine_sentiment' in line['label']:
			label_flu_vaccine_sentiment.append(line['label']['flu_vaccine_sentiment'])
		else:
			label_flu_vaccine_sentiment.append(None)

	return data,label_flu_vaccine_intent_to_receive,label_flu_vaccine_received,\
	label_flu_vaccine_relevant,label_flu_vaccine_sentiment


#handle health.json.gz.json file
def handle_health_json(path):
	json_data=[]
	with gzip.open(path, 'rb') as f:
		file_content = f.readlines()
		for line in file_content:
			json_data.append(json.loads(line.decode("utf-8")))


	data=[]
	label=[]
	for line in json_data:
		data.append(line['tweet']['text'])
		label.append(line['label']['health_relevant'])
	return data,label

# handle the data for trust-in-government.json
def handle_trust_in_gov(path):
	json_data=[]
	with gzip.open(path, 'rb') as f:
		file_content = f.readlines()
		for line in file_content:
			json_data.append(json.loads(line.decode("utf-8")))

	data=[]
	label_about_gov=[]
	label_about_vaccine=[]
	label_trust_gov=[]
	for line in json_data:
		data.append(line['text'])
		label_about_gov.append(line['label']['about_gov'])
		label_about_vaccine.append(line['label']['about_vaccines'])
		label_trust_gov.append(line['label']['trust_gov'])
	return data,label_about_gov,label_about_vaccine,label_trust_gov

#handle data of vaccine_sentiment.json
def handle_vaccine_sentiment(path):
	json_data=[]
	with gzip.open(path, 'rb') as f:
		file_content = f.readlines()
		for line in file_content:
			json_data.append(json.loads(line.decode("utf-8")))

	data=[]
	label_relevant=[]

	for line in json_data:
		data.append(line['text'])
		label_relevant.append(line['label']['relevant'])
	return data,label_relevant

#handle the data of zika_conspiracy.json
def handle_zika_conspiracy(path):
	json_data=[]
	with gzip.open(path, 'rb') as f:
		file_content = f.readlines()
		for line in file_content:
			json_data.append(json.loads(line.decode("utf-8")))


	data=[]
	labels=[]
	for line in json_data:
		data.append(line['tweet']['text'])
		labels.append(line['label']['zika_conspiracy'])
	return data,labels

