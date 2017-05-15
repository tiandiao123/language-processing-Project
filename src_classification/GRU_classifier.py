import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from pprint import pprint
from process_data import handle_flu_json,handle_flu_risk_perception,handle_flu_vaccine_new
from process_data import handle_health_json,handle_trust_in_gov,handle_vaccine_sentiment,handle_zika_conspiracy
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from Tokenizer_text import tokenize_words
import time
#import Tokenizer_text
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score
import nltk



vocabulary_size=8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token="SENTENCE_START_TOKEN"



def tokenize_words(txt,labels):
	sentences=[]
	for i in range(len(txt)):
		sentences.append("{} {}".format(sentence_start_token,txt[i]))
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(sentences)
	sequences = tokenizer.texts_to_sequences(sentences)
	label_index=tokenizer.texts_to_sequences(labels)
	return sequences,label_index

def GRU_train_prediction(sequences,labels,max_review_length,top_words,num_iterations):
	#top_words = 10000
	labels=np.array(labels)
	label_array=labels.flatten()
        
	label2index=list(set(label_array))
	#print(label2index)
	prediction_labels=[]
	
	for i in range(len(labels)):
		if labels[i]==label2index[0]:
			prediction_labels.append(0)
		else:
			prediction_labels.append(1)
	#print(prediction_labels)
        
	prediction_labels=np.array(prediction_labels)
	X_train,X_test,y_train,y_test=train_test_split(sequences,prediction_labels,test_size=0.3)
	#max_review_length = 500
	X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
	X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

	X_train=np.array(X_train)
	X_test=np.array(X_test)

	y_train_c = keras.utils.to_categorical(y_train, 2)
	y_test_c = keras.utils.to_categorical(y_test, 2)
	
	# print(X_train.shape)
	# print(X_test.shape)

	# print(y_test)
	# print(y_test_c)

	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
	model.add(Dropout(0.2))
	model.add(GRU(30))
	model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(X_train, y_train_c, validation_data=(X_test, y_test_c), epochs=num_iterations, batch_size=64)
	scores = model.evaluate(X_test, y_test_c, verbose=0)
	test_acc=scores[1]
	
	probs = model.predict(x=X_test)
	# print(probs)
	predics=[]
	for ele in probs:
	 	if ele[1]>=0.5:
	 		predics.append(1)
	 	else:
	 		predics.append(0)

	#print(predics)

	test_Recall=recall_score(y_test, predics, average='macro') 
	test_Precision= precision_score(y_test, predics, average='macro')
	test_F_1=2*(test_Recall*test_Precision)/(test_Precision+test_Recall)
	test_auc=roc_auc_score(y_test,probs[:,1])

	return test_acc,test_F_1,test_Recall,test_Precision,test_auc

f=open("GRU_classifer_performance.txt","w+")
f.write("Dataset | Task | Model | FeatureSet | EvaluationSet | Accuracy | Precision | Recall | F1 Score | AUC\n")


txt_flu,labels_flu=handle_flu_json("../data/flu.json.gz")
sequences,labels =tokenize_words(txt_flu,labels_flu)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,10000,3)

f.write("flu | flu_relevant | GRU_classifier | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))



train_data,label_about_flu,label_about_fluShot,label_about_flu_likelihood,label_about_flu_severity=\
handle_flu_risk_perception("../data/flu-risk-perception.json.gz")

#case 1:
txt_about_flu=[]
prediction_labels_about_flu_likelihood=[]
for i in range(len(label_about_flu_likelihood)):
    if label_about_flu_likelihood[i]=='likely':
        prediction_labels_about_flu_likelihood.append("likely")
        txt_about_flu.append(train_data[i])
    elif label_about_flu_likelihood[i]=='unlikely':
        prediction_labels_about_flu_likelihood.append("unlikely")
        txt_about_flu.append(train_data[i])

sequences,labels=tokenize_words(txt_about_flu,prediction_labels_about_flu_likelihood)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,10000,5)

f.write("flu-risk | label_about_flu_likelihood | GRU_classifier | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))


#case 2:
txt_about_flu=[]
prediction_labels_about_flu=[]
for i in range(len(label_about_flu)):
    if label_about_flu[i]=='yes':
        prediction_labels_about_flu.append("yes")
        txt_about_flu.append(train_data[i])
    elif label_about_flu[i]=='no':
        prediction_labels_about_flu.append("no")
        txt_about_flu.append(train_data[i])

print(len(txt_about_flu))

sequences,labels=tokenize_words(txt_about_flu,prediction_labels_about_flu)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,18000,3)

f.write("flu-risk | label_about_flu | GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))




txt_about_flu=[]
prediction_labels_about_flushot=[]
for i in range(len(label_about_fluShot)):
    if label_about_fluShot[i]=='yes':
        prediction_labels_about_flushot.append("yes")
        txt_about_flu.append(train_data[i])
    elif label_about_fluShot[i]=='no':
        prediction_labels_about_flushot.append("no")
        txt_about_flu.append(train_data[i])


print(len(txt_about_flu))

sequences,labels=tokenize_words(txt_about_flu,prediction_labels_about_flushot)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,18000,3)

f.write("flu-risk | label_about_flushot | GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))





data,label_flu_vaccine_intent_to_receive,label_flu_vaccine_received,\
    label_flu_vaccine_relevant,label_flu_vaccine_sentiment=handle_flu_vaccine_new("../data/flu_vaccine.json.gz")

txt=[]
prediction_labels_intend=[]
for i in range(len(label_flu_vaccine_intent_to_receive)):
    if label_flu_vaccine_intent_to_receive[i]=='yes':
        prediction_labels_intend.append("yes")
        txt.append(data[i])
    elif label_flu_vaccine_intent_to_receive[i]=='no':
        prediction_labels_intend.append("no")
        txt.append(data[i])

print(len(txt))

sequences,labels=tokenize_words(txt,prediction_labels_intend)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,19000,3)

f.write("flu-vaccine | label_flu_vaccine_intent_to_receive | GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))


txt=[]
prediction_labels_flu_vaccine_relevant=[]
for i in range(len(label_flu_vaccine_relevant)):
    if label_flu_vaccine_relevant[i]=='yes':
        prediction_labels_flu_vaccine_relevant.append("yes")
        txt.append(data[i])
    elif label_flu_vaccine_relevant[i]=='no':
        prediction_labels_flu_vaccine_relevant.append("no")
        txt.append(data[i])


print(len(txt))

sequences,labels=tokenize_words(txt,prediction_labels_flu_vaccine_relevant)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,19000,3)

f.write("flu-vaccine | prediction_labels_flu_vaccine_relevant | GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))



data,labels=handle_health_json("../data/health.json.gz")

txt=[]
prediction_labels=[]
for i in range(len(labels)):
    if labels[i]=='health':
        prediction_labels.append("health")
        txt.append(data[i])
    elif labels[i]=='sick':
        prediction_labels.append("sick")
        txt.append(data[i])


print(len(txt))

sequences,labels=tokenize_words(txt,prediction_labels)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,19000,3)

f.write("health.json | health-sick | GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))



data,label_about_gov,label_about_vaccine,label_trust_gov=\
handle_trust_in_gov('../data/trust-in-government.json.gz')



txt=[]
prediction_labels=[]
for i in range(len(label_trust_gov)):
    if label_trust_gov[i]=='yes_trust':
        prediction_labels.append("yes")
        txt.append(data[i])
    elif label_trust_gov[i]=='neither_trust':
        prediction_labels.append("neither")
        txt.append(data[i])

txt=np.array(txt)
prediction_labels=np.array(prediction_labels)

print(len(txt))

sequences,labels=tokenize_words(txt,prediction_labels)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,19000,3)

f.write("trust-in-government | label_trust_gov| GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))


txt=[]
prediction_labels=[]
for i in range(len(label_about_gov)):
    if label_about_gov[i]=='yes':
        prediction_labels.append("yes")
        txt.append(data[i])
    elif label_about_gov[i]=='no':
        prediction_labels.append("no")
        txt.append(data[i])

print(len(txt))

sequences,labels=tokenize_words(txt,prediction_labels)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,19000,3)

f.write("trust-in-government | label_about_gov| GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))



txt=[]
prediction_labels=[]
for i in range(len(label_about_vaccine)):
    if label_about_vaccine[i]=='yes':
        prediction_labels.append("yes")
        txt.append(data[i])
    elif label_about_vaccine[i]=='no':
        prediction_labels.append("no")
        txt.append(data[i])



print(len(txt))

sequences,labels=tokenize_words(txt,prediction_labels)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,19000,3)

f.write("trust-in-government | label_about_vaccine| GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))



data,label_relevant=handle_vaccine_sentiment("../data/vaccine_sentiment.json.gz")

txt=[]
prediction_labels=[]
for i in range(len(label_relevant)):
    if label_relevant[i]=='yes':
        prediction_labels.append("yes")
        txt.append(data[i])
    elif label_relevant[i]=='no':
        prediction_labels.append("no")
        txt.append(data[i])



print(len(txt))

sequences,labels=tokenize_words(txt,prediction_labels)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,19000,3)

f.write("vaccine_sentiment | yes-no| GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))







