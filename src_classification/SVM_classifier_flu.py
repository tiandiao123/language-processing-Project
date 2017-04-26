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
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc



txt,labels=handle_flu_json("../data/flu.json.gz")
prediction_labels=[]
for ele in labels:
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


print("use svm training flurate(C=1)>>>")
start=time.time()
#construct SVM model, and compuet dev accuracy, precision, recall, F1 and auc value
dev_acc=[]
dev_auc=[]
dev_precision=[]
dev_recall=[]
dev_F1=[]
clf = linear_model.SGDClassifier()
i=0
for train_index, dev_index in kf.split(X_Kfold):
    print("the {} iteraion".format(str(i)))
    X_validation=X_Kfold[dev_index]
    y_validation=y_Kfold[dev_index]
    X_train=X_Kfold[train_index]
    y_train=y_Kfold[train_index]
    #print(len(X_train))
    #print(len(y_train))
    clf.fit(X_train,y_train)
    predics=clf.predict(X_validation)

    y_validation_scores=clf.decision_function(X_validation)
    fpr, tpr, _ = roc_curve(y_validation, y_validation_scores)
    roc_auc = auc(fpr, tpr)
    dev_auc.append(roc_auc)

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
print("flu | flu_relevant | linear_svm | unigram | dev | {} | {} | {} | {} | {}"\
	.format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),str(np.mean(dev_auc))))






print("evaluation test_data......")
predics=clf.predict(X_test)

y_test_scores=clf.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_test_scores)
auc = auc(fpr, tpr)


num_correct=np.sum(np.array(predics)==y_test)
### compute precison, recall,auc value
Recall=recall_score(y_test, predics, average='macro') 
Precision= precision_score(y_test, predics, average='macro') 
test_accracy="{:.3f}".format(float(num_correct)/len(y_test))
F_1=2*(Recall*Precision)/(Precision+Recall)

print("flu | flu_relevant | linear_svm | unigram | test | {} | {} | {} | {} | {}"\
	.format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))
