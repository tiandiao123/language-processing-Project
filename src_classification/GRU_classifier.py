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
import Tokenizer_text
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score


def GRU_train_prediction(sequences,labels,max_review_length,top_words):
	#split the data into training data and test data
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
	X_train,X_test,y_train,y_test=train_test_split(sequences,prediction_labels,test_size=0.2)
	#max_review_length = 500
	X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
	X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

	X_train=np.array(X_train)
	X_test=np.array(X_test)

	y_train_c = keras.utils.to_categorical(y_train, 2)
	y_test_c = keras.utils.to_categorical(y_test, 2)
	
	print(X_train.shape)
	print(X_test.shape)

	print(y_test)
	#print(y_test_c)

	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
	model.add(Dropout(0.2))
	model.add(GRU(30))
	model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(X_train, y_train_c, validation_data=(X_test, y_test_c), epochs=3, batch_size=64)
	scores = model.evaluate(X_test, y_test_c, verbose=0)
	test_acc=scores[1]
	
	probs = model.predict(x=X_test)
	print(probs)
	predics=[]
	for ele in probs:
	 	if ele[1]>=0.5:
	 		predics.append(1)
	 	else:
	 		predics.append(0)

	print(predics)

	test_Recall=recall_score(y_test, predics, average='macro') 
	test_Precision= precision_score(y_test, predics, average='macro')
	test_F_1=2*(test_Recall*test_Precision)/(test_Precision+test_Recall)
	test_auc=roc_auc_score(y_test,probs[:,1])

	# print("Testing Accuracy: %.2f%%" % (scores[1]*100))
	# print("F_1 for testing data is : {}".format(str(F_1)))
	# print("Recall for testing data is : {}".format(str(Recall)))
	# print("Precision for testing data is : {}".format(str(Precision)))
	# print("the auc value for testing data is :{}".format(str(auc)))
	return test_acc,test_F_1,test_Recall,test_Precision,test_auc
	

# f=open("GRU_classifer_performance.txt","w+")
# f.write("Dataset | Task | Model | FeatureSet | EvaluationSet | Accuracy | Precision | Recall | F1 Score | AUC\n")


txt_flu,labels_flu=handle_flu_json("../data/flu.json.gz")
sequences,labels=tokenize_words(txt_flu,labels_flu)
test_acc,test_F_1,test_Recall,test_Precision,test_auc=GRU_train_prediction(sequences,labels,40,10000)

print("flu | flu_relevant | GRU_RNN | None | test | %.6f%% | %.6f%% | %.6f%% | %.6f%% | %.6f%% \n" % \
	(test_auc*100,test_Precision*100,test_Recall*100,test_F_1*100,test_auc*100))



##################################################################################################################

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



