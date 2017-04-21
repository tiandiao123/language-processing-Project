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


# In[51]:

txt,labels=handle_flu_json("../json_data/flu.json")
prediction_labels=[]
for ele in labels:
    if ele=='true':
        prediction_labels.append(1)
    else:
        prediction_labels.append(0)
prediction_labels=np.array(prediction_labels)


# In[52]:

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


# In[53]:

kf = KFold(n_splits=4)
y_Kfold[0:5]


# In[56]:

print("use svm training flurate(set C=0.001)>>>")
start=time.time()
#construct SVM model
acc=[]
clf = svm.SVC(C=0.001,kernel='linear')
clf.fit(X_Kfold,y_Kfold)
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
    num_corrects=np.sum(np.array(predics)==np.array(y_validation))
    i+=1
    acc.append(float(num_corrects)/len(y_validation))
print("the average validation accuracy for K-fold(C=0.001) is : ",np.mean(acc))


print("use svm training flurate(set C=0.01)>>>")
start=time.time()
#construct SVM model
acc=[]
clf = svm.SVC(C=0.001,kernel='linear')
clf.fit(X_Kfold,y_Kfold)
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
    num_corrects=np.sum(np.array(predics)==np.array(y_validation))
    i+=1
    acc.append(float(num_corrects)/len(y_validation))
print("the average validation accuracy for K-fold(C=0.01) is : ",np.mean(acc))


print("use svm training flurate(set C=0.1)>>>")
start=time.time()
#construct SVM model
acc=[]
clf = svm.SVC(C=0.001,kernel='linear')
clf.fit(X_Kfold,y_Kfold)
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
    num_corrects=np.sum(np.array(predics)==np.array(y_validation))
    i+=1
    acc.append(float(num_corrects)/len(y_validation))
print("the average validation accuracy for K-fold(C=0.1) is : ",np.mean(acc))


print("use svm training flurate(set C=1)>>>")
start=time.time()
#construct SVM model
acc=[]
clf = svm.SVC(C=0.001,kernel='linear')
clf.fit(X_Kfold,y_Kfold)
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
    num_corrects=np.sum(np.array(predics)==np.array(y_validation))
    i+=1
    acc.append(float(num_corrects)/len(y_validation))
print("the average validation accuracy for K-fold(C=1) is : ",np.mean(acc))



print("use svm training flurate(set C=10)>>>")
start=time.time()
#construct SVM model
acc=[]
clf = svm.SVC(C=0.001,kernel='linear')
clf.fit(X_Kfold,y_Kfold)
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
    num_corrects=np.sum(np.array(predics)==np.array(y_validation))
    i+=1
    acc.append(float(num_corrects)/len(y_validation))
print("the average validation accuracy for K-fold(C=10) is : ",np.mean(acc))