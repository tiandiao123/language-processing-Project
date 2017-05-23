import numpy as np
import sys
from statistics import mean
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
#from Tokenizer_text import tokenize_words
import time
#import Tokenizer_text
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score



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
	return sequences

def cross_validation_model_selection(l2_parameters,sequences,labels,max_review_length,\
	top_words,num_iterations):
	kf = KFold(n_splits=4)
	optimal_para=l2_parameters[0]

	dev_accuracy=sys.maxsize
	dev_precision=0
	dev_recall=0
	dev_F_1=0
	dev_auc=0

	sequences=np.array(sequences)
	labels=np.array(labels)

	X_dev_sequences, X_test_sequences, y_dev_labels, y_test_labes = train_test_split(sequences, labels,\
	 test_size=0.2, random_state=42)

	for l2_val in l2_parameters:
		loss_list=[]
		recall_list=[]
		precision_list=[]
		auc_list=[]
		F1_list=[]
		fold=0
		print("testing parameters {}".format(str(l2_val)))

		for train_index, test_index in kf.split(X_dev_sequences):
			fold+=1
			print("evaluating fold {}".format(str(fold)))
			X_train, X_test = X_dev_sequences[train_index], X_dev_sequences[test_index]
			y_train, y_test = y_dev_labels[train_index], y_dev_labels[test_index]

			X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
			X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
			X_train=np.array(X_train)
			X_test=np.array(X_test)
			y_train_c = keras.utils.to_categorical(y_train, 2)
			y_test_c = keras.utils.to_categorical(y_test, 2)


			embedding_vecor_length = 32
			model = Sequential()
			model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
			#model.add(Dropout(0.2))
			model.add(GRU(50,kernel_regularizer=keras.regularizers.l2(0.01),bias_regularizer=keras.regularizers.l2(0.01)))
			model.add(Dropout(0.3))
			model.add(Dense(10,activation='relu'))
			model.add(Dense(2, activation='softmax'))
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			#print(model.summary()),
			model.fit(X_train, y_train_c, validation_data=(X_test, y_test_c), epochs=num_iterations, batch_size=64)
			scores = model.evaluate(X_test, y_test_c, verbose=0)
			test_acc=scores[1]
			loss_list.append(test_acc)

			probs = model.predict(x=X_test)
			predics=[]
			for ele in probs:
			 	if ele[1]>=0.5:
			 		predics.append(1)
			 	else:
			 		predics.append(0)


			test_Recall=recall_score(y_test, predics, average='macro')
			recall_list.append(test_Recall) 
			test_Precision= precision_score(y_test, predics, average='macro')
			precision_list.append(test_Precision)
			test_F_1=2*(test_Recall*test_Precision)/(test_Precision+test_Recall)
			F1_list.append(test_F_1)
			test_auc=roc_auc_score(y_test,probs[:,1])
			auc_list.append(test_auc)

		average_loss=mean(loss_list)
		if dev_accuracy>average_loss:

			dev_accuracy=average_loss
			optimal_para=l2_val
			dev_precision=mean(precision_list)
			dev_recall=mean(recall_list)
			dev_auc=mean(auc_list)
			dev_F_1=mean(F1_list)

	print("model has been selected, begin apply it in testing data")
	test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
	GRU_train_prediction(sequences,labels,max_review_length,top_words,num_iterations,optimal_para)

	return dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc









def GRU_train_prediction(sequences,labels,max_review_length,top_words,num_iterations,oprimal_para):
        
	prediction_labels=np.array(labels)
	X_train,X_test,y_train,y_test = train_test_split(sequences, prediction_labels,test_size=0.2, random_state=42)
	#max_review_length = 500
	X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
	X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

	X_train=np.array(X_train)
	X_test=np.array(X_test)

	y_train_c = keras.utils.to_categorical(y_train, 2)
	y_test_c = keras.utils.to_categorical(y_test, 2)
	
	# print(X_train.shape)
	# print(X_test.shape)

	#print(y_test)
	#print(y_test_c)

	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
	#model.add(Dropout(0.2))
	model.add(GRU(50,bias_regularizer=keras.regularizers.l2(oprimal_para)))
	model.add(Dropout(0.3))
	model.add(Dense(10,activation='relu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary()),
	model.fit(X_train, y_train_c, validation_data=(X_test, y_test_c), epochs=num_iterations, batch_size=64)
	scores = model.evaluate(X_test, y_test_c, verbose=0)
	test_acc=scores[1]
	
	probs = model.predict(x=X_test)
	#print(probs)

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
sequences=tokenize_words(txt_flu,labels_flu)

prediction_labels_flu=[]
for ele in labels_flu:
    if ele=='true':
        prediction_labels_flu.append(1)
    else:
        prediction_labels_flu.append(0)

dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
cross_validation_model_selection([0.0001,0.001,0.01,0.1,1],sequences,prediction_labels_flu,40,10000,3)

f.write("flu | flu_relevant | GRU_classifier | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(dev_accuracy),str(dev_precision),str(dev_recall),str(dev_F_1),str(dev_auc)))

f.write("flu | flu_relevant | GRU_classifier | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))




train_data,label_about_flu,label_about_fluShot,label_about_flu_likelihood,label_about_flu_severity=\
handle_flu_risk_perception("../data/flu-risk-perception.json.gz")

#case 1:
txt_about_flu=[]
prediction_labels_about_flu_likelihood=[]

for i in range(len(label_about_flu_likelihood)):
    if label_about_flu_likelihood[i]=='likely':
        prediction_labels_about_flu_likelihood.append(1)
        txt_about_flu.append(train_data[i])
    elif label_about_flu_likelihood[i]=='unlikely':
        prediction_labels_about_flu_likelihood.append(0)
        txt_about_flu.append(train_data[i])

sequences=tokenize_words(txt_about_flu,prediction_labels_about_flu_likelihood)

dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
cross_validation_model_selection([0.0001,0.001,0.01,0.1,1],sequences,prediction_labels_about_flu_likelihood,\
	40,10000,5)

f.write("flu-risk | label_about_flu_likelihood | GRU_classifier | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(dev_accuracy),str(dev_precision),str(dev_recall),str(dev_F_1),str(dev_auc)))
f.write("flu-risk | label_about_flu_likelihood | GRU_classifier | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))


#case 2:
txt_about_flu=[]
prediction_labels_about_flu=[]
for i in range(len(label_about_flu)):
    if label_about_flu[i]=='yes':
        prediction_labels_about_flu.append(1)
        txt_about_flu.append(train_data[i])
    elif label_about_flu[i]=='no':
        prediction_labels_about_flu.append(0)
        txt_about_flu.append(train_data[i])

sequences=tokenize_words(txt_about_flu,prediction_labels_about_flu)

dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
cross_validation_model_selection([0.0001,0.001,0.01,0.1,1],sequences,prediction_labels_about_flu,40,18000,5)

f.write("flu-risk | label_about_flu | GRU_classifer | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(dev_accuracy),str(dev_precision),str(dev_recall),str(dev_F_1),str(dev_auc)))
f.write("flu-risk | label_about_flu | GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))




txt_about_flu=[]
prediction_labels_about_flushot=[]
for i in range(len(label_about_fluShot)):
    if label_about_fluShot[i]=='yes':
        prediction_labels_about_flushot.append(1)
        txt_about_flu.append(train_data[i])
    elif label_about_fluShot[i]=='no':
        prediction_labels_about_flushot.append(0)
        txt_about_flu.append(train_data[i])

sequences=tokenize_words(txt_about_flu,prediction_labels_about_flushot)
dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
cross_validation_model_selection([0.0001,0.001,0.01,0.1,1],sequences,prediction_labels_about_flushot,40,18000,5)

f.write("flu-risk | label_about_flushot | GRU_classifer | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(dev_accuracy),str(dev_precision),str(dev_recall),str(dev_F_1),str(dev_auc)))
f.write("flu-risk | label_about_flushot | GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))





data,label_flu_vaccine_intent_to_receive,label_flu_vaccine_received,\
    label_flu_vaccine_relevant,label_flu_vaccine_sentiment=handle_flu_vaccine_new("../data/flu_vaccine.json.gz")

txt=[]
prediction_labels_intend=[]
for i in range(len(label_flu_vaccine_intent_to_receive)):
    if label_flu_vaccine_intent_to_receive[i]=='yes':
        prediction_labels_intend.append(1)
        txt.append(data[i])
    elif label_flu_vaccine_intent_to_receive[i]=='no':
        prediction_labels_intend.append(0)
        txt.append(data[i])

sequences=tokenize_words(txt,prediction_labels_intend)
dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
cross_validation_model_selection([0.0001,0.001,0.01,0.1,1],sequences,prediction_labels_intend,40,19000,5)

f.write("flu-vaccine | label_flu_vaccine_intent_to_receive | GRU_classifer | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(dev_accuracy),str(dev_precision),str(dev_recall),str(dev_F_1),str(dev_auc)))
f.write("flu-vaccine | label_flu_vaccine_intent_to_receive | GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))


txt=[]
prediction_labels_flu_vaccine_relevant=[]
for i in range(len(label_flu_vaccine_relevant)):
    if label_flu_vaccine_relevant[i]=='yes':
        prediction_labels_flu_vaccine_relevant.append(1)
        txt.append(data[i])
    elif label_flu_vaccine_relevant[i]=='no':
        prediction_labels_flu_vaccine_relevant.append(0)
        txt.append(data[i])


# print(len(txt))

sequences=tokenize_words(txt,prediction_labels_flu_vaccine_relevant)

dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
cross_validation_model_selection([0.0001,0.001,0.01,0.1,1],sequences,prediction_labels_flu_vaccine_relevant,\
	40,19000,5)


f.write("flu-vaccine | prediction_labels_flu_vaccine_relevant | GRU_classifer | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(dev_accuracy),str(dev_precision),str(dev_recall),str(dev_F_1),str(dev_auc)))
f.write("flu-vaccine | prediction_labels_flu_vaccine_relevant | GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))



data,labels=handle_health_json("../data/health.json.gz")

txt=[]
prediction_labels=[]
for i in range(len(labels)):
    if labels[i]=='health':
        prediction_labels.append(1)
        txt.append(data[i])
    elif labels[i]=='sick':
        prediction_labels.append(0)
        txt.append(data[i])


print(len(txt))

sequences=tokenize_words(txt,prediction_labels)
dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
cross_validation_model_selection([0.0001,0.001,0.01,0.1,1],sequences,prediction_labels,40,19000,5)

f.write("health.json | health-sick | GRU_classifer | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(dev_accuracy),str(dev_precision),str(dev_recall),str(dev_F_1),str(dev_auc)))
f.write("health.json | health-sick | GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))



data,label_about_gov,label_about_vaccine,label_trust_gov=\
handle_trust_in_gov('../data/trust-in-government.json.gz')



txt=[]
prediction_labels=[]
for i in range(len(label_trust_gov)):
    if label_trust_gov[i]=='yes_trust':
        prediction_labels.append(1)
        txt.append(data[i])
    elif label_trust_gov[i]=='neither_trust':
        prediction_labels.append(0)
        txt.append(data[i])

txt=np.array(txt)
prediction_labels=np.array(prediction_labels)
sequences=tokenize_words(txt,prediction_labels)

dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
cross_validation_model_selection([0.0001,0.001,0.01,0.1,1],sequences,prediction_labels,40,19000,5)


f.write("trust-in-government | label_trust_gov| GRU_classifer | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(dev_accuracy),str(dev_precision),str(dev_recall),str(dev_F_1),str(dev_auc)))
f.write("trust-in-government | label_trust_gov| GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))


txt=[]
prediction_labels=[]
for i in range(len(label_about_gov)):
    if label_about_gov[i]=='yes':
        prediction_labels.append(1)
        txt.append(data[i])
    elif label_about_gov[i]=='no':
        prediction_labels.append(0)
        txt.append(data[i])

print(len(txt))

sequences=tokenize_words(txt,prediction_labels)
dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
cross_validation_model_selection([0.0001,0.001,0.01,0.1,1],sequences,prediction_labels,40,19000,5)


f.write("trust-in-government | label_about_gov| GRU_classifer | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(dev_accuracy),str(dev_precision),str(dev_recall),str(dev_F_1),str(dev_auc)))
f.write("trust-in-government | label_about_gov| GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))



txt=[]
prediction_labels=[]
for i in range(len(label_about_vaccine)):
    if label_about_vaccine[i]=='yes':
        prediction_labels.append(1)
        txt.append(data[i])
    elif label_about_vaccine[i]=='no':
        prediction_labels.append(0)
        txt.append(data[i])



print(len(txt))

sequences=tokenize_words(txt,prediction_labels)
dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
cross_validation_model_selection([0.0001,0.001,0.01,0.1,1],sequences,prediction_labels,40,19000,5)

f.write("trust-in-government | label_about_vaccine| GRU_classifer | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(dev_accuracy),str(dev_precision),str(dev_recall),str(dev_F_1),str(dev_auc)))
f.write("trust-in-government | label_about_vaccine| GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))



data,label_relevant=handle_vaccine_sentiment("../data/vaccine_sentiment.json.gz")

txt=[]
prediction_labels=[]
for i in range(len(label_relevant)):
    if label_relevant[i]=='yes':
        prediction_labels.append(1)
        txt.append(data[i])
    elif label_relevant[i]=='no':
        prediction_labels.append(0)
        txt.append(data[i])



print(len(txt))

sequences=tokenize_words(txt,prediction_labels)
dev_accuracy,dev_F_1,dev_recall,dev_precision,dev_auc,test_acc,test_F_1,test_Recall,test_Precision,test_auc=\
cross_validation_model_selection([0.0001,0.001,0.01,0.1,1],sequences,prediction_labels,40,19000,5)


f.write("vaccine_sentiment | label_relavant| GRU_classifer | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(dev_accuracy),str(dev_precision),str(dev_recall),str(dev_F_1),str(dev_auc)))
f.write("vaccine_sentiment | label_relavant| GRU_classifer | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_acc),str(test_Precision),str(test_Recall),str(test_F_1),str(test_auc)))


