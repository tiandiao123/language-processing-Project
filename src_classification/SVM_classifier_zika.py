#!/usr/bin/env python
import numpy as np
import json
from pprint import pprint
import time
import math
from textblob import TextBlob as tb
from sklearn.model_selection import train_test_split
from sklearn import svm
import sklearn.svm
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from process_data import handle_flu_json,handle_flu_risk_perception
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

txtstr,labels=handle_flu_json("../../json_data/zika_conspiracy.json")

#pprint(labels)
pprint(len(labels))
txt=[]
prediction_labels=[]
for i in range(len(txtstr)):
	if txtstr[i]!='\n' and txtstr[i]!='0':
		txt.append(txtstr[i])
		if ele=='true':
			prediction_labels.append(1)
		else:
			prediction_labels.append(0)
prediction_labels=np.array(prediction_labels)


## make tf_idf_matrix
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(txt)
freq_term_matrix=count_vectorizer.transform(txt)

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)

tf_idf_matrix = tfidf.transform(freq_term_matrix)
data=tf_idf_matrix.todense()

#split the data into training data and test data
X_Kfold,X_test,y_Kfold,y_test=train_test_split(data,prediction_labels,test_size=0.33)
kf = KFold(n_splits=4)


print("use svm training zika_conspiracy(C=1)>>>")
start=time.time()
#construct SVM model, and compuet dev accuracy, precision, recall, F1 and auc value
dev_acc=[]
dev_auc=[]
dev_precision=[]
dev_recall=[]
dev_F1=[]
clf = svm.SVC(C=1,kernel='linear',probability=True)
i=0
for train_index, dev_index in kf.split(X_Kfold):
    print("the {} iteraion".format(str(i)))
    X_validation=X_Kfold[dev_index]
    y_validation=y_Kfold[dev_index]
    X_train=X_Kfold[train_index]
    y_train=y_Kfold[train_index]
    print(len(X_train))
    print(len(y_train))
    clf.fit(X_train,y_train)
    predics=clf.predict(X_validation)
    predics_prob=clf.predict_proba(X_validation)
    probs=[]
    for j in range(len(y_validation)):
    	if y_validation[j]==1:
    		probs.append(predics_prob[j][1])
    	else:
    		probs.append(predics_prob[j][0])
    dev_auc.append(roc_auc_score(y_validation,probs))
    num_corrects=np.sum(np.array(predics)==np.array(y_validation))
    i+=1
    dev_acc.append(float(num_corrects)/len(y_validation))
    recall=recall_score(y_validation,predics,average='macro')
    precision=precision_score(y_validation,predics,average='macro')
    F_1=2*(recall*precision)/(recall+precision)
    dev_F1.append(F_1)
    dev_precision.append(precision)
    dev_recall.append(recall)
end=time.time()
print("the training time is {}".format(str(end-start)))
print("zika | zika_conspiracy | linear_svm | unigram | dev | {} | {} | {} | {} | {}"\
	.format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),str(np.mean(dev_auc))))






print("evaluation test_data......")
predics_prob=clf.predict_proba(X_test)
probs=[]
for i in range(len(y_test)):
    if y_test[i]==1:
        probs.append(predics_prob[i][1])
    else:
        probs.append(predics_prob[i][0])

auc=roc_auc_score(y_test,probs)

predics=[]
for i in range(len(predics_prob)):
	if predics_prob[i][0]>=predics_prob[i][1]:
		predics.append(0)
	else:
		predics.append(1)


num_correct=np.sum(np.array(predics)==y_test)
### compute precison, recall,auc value
Recall=recall_score(y_test, predics, average='macro') 
Precision= precision_score(y_test, predics, average='macro') 
test_accracy="{:.3f}".format(float(num_correct)/len(y_test))
F_1=2*(Recall*Precision)/(Precision+Recall)

print("zika | zika_conspiracy | linear_svm | unigram | test | {} | {} | {} | {} | {}"\
	.format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))