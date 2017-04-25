#!/usr/bin/env python
import numpy as np
import json
from pprint import pprint
import time
import math
from sklearn.model_selection import train_test_split
from sklearn import svm
import sklearn.svm
from sklearn.dummy import DummyClassifier
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from process_data import handle_flu_json,handle_flu_risk_perception
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def handle_two_classifications(txt,prediction_labels):
    ##make td-idf matric
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


    #print("use svm training flurate(C=1)>>>")
    start=time.time()
    #construct SVM model, and compuet dev accuracy, precision, recall, F1 and auc value
    dev_acc=[]
    dev_auc=[]
    dev_precision=[]
    dev_recall=[]
    dev_F1=[]
    clf = DummyClassifier(strategy='most_frequent',random_state=0)
    i=0
    for train_index, dev_index in kf.split(X_Kfold):
        #print("the {} iteraion".format(str(i)))
        X_validation=X_Kfold[dev_index]
        y_validation=y_Kfold[dev_index]
        X_train=X_Kfold[train_index]
        y_train=y_Kfold[train_index]
        #print(len(X_train))
        #print(len(y_train))
        clf.fit(X_train,y_train)
        predics=clf.predict(X_validation)
        predics_prob=clf.predict_proba(X_validation)
        probs = predics_prob[:,1]
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
    #print("the training time is {}".format(str(end-start)))


    print("evaluation test_data......")
    predics_prob=clf.predict_proba(X_test)
    probs = predics_prob[:,1]
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
    return dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc


txt_flu,labels_flu=handle_flu_json("../data/flu.json.gz")
prediction_labels_flu=[]
for ele in labels_flu:
    if ele=='true':
        prediction_labels_flu.append(1)
    else:
        prediction_labels_flu.append(0)

prediction_labels_flu=np.array(prediction_labels_flu)


dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt_flu,prediction_labels_flu)

print("flu | flu_relevant | majority | None | dev | {} | {} | {} | {} | {}"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

print("flu | flu_relevant | majority | None | test | {} | {} | {} | {} | {}"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))


train_data,label_about_flu,label_about_fluShot,label_about_flu_likelihood,label_about_flu_severity=\
handle_flu_risk_perception("../data/flu-risk-perception.json.gz")


#case 1:
txt_about_flu=[]
prediction_labels_about_flu_likelihood=[]
for ele in label_about_flu_likelihood:
    if ele=='likely':
        prediction_labels_about_flu_likelihood.append(1)
        txt_about_flu.append(ele)
    elif ele=='unlikely':
        prediction_labels_about_flu_likelihood.append(0)
        txt_about_flu.append(ele)

txt_about_flu=np.array(txt_about_flu)
prediction_labels_about_flu_likelihood=np.array(prediction_labels_about_flu_likelihood)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt_about_flu,prediction_labels_about_flu_likelihood)

print("flu-risk | label_about_flu_likelihood | majority | None | dev | {} | {} | {} | {} | {}"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

print("flu-risk | label_about_flu_likelihood | majority | None | test | {} | {} | {} | {} | {}"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))

#case2:
txt_about_flu=[]
prediction_labels_about_flu=[]
for ele in label_about_flu:
    if ele=='yes':
        prediction_labels_about_flu.append(1)
        txt_about_flu.append(ele)
    elif ele=='no':
        prediction_labels_about_flu.append(0)
        txt_about_flu.append(ele)

txt_about_flu=np.array(txt_about_flu)
prediction_labels_about_flu=np.array(prediction_labels_about_flu)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt_about_flu,prediction_labels_about_flu)

print("flu-risk | label_about_flu | majority | None | dev | {} | {} | {} | {} | {}"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

print("flu-risk | label_about_flu | majority | None | test | {} | {} | {} | {} | {}"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))

#case3:
txt_about_flu=[]
prediction_labels_about_flushot=[]
for ele in label_about_fluShot:
    if ele=='yes':
        prediction_labels_about_flushot.append(1)
        txt_about_flu.append(ele)
    elif ele=='no':
        prediction_labels_about_flushot.append(0)
        txt_about_flu.append(ele)

txt_about_flu=np.array(txt_about_flu)
prediction_labels_about_flushot=np.array(prediction_labels_about_flushot)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt_about_flu,prediction_labels_about_flushot)

print("flu-risk | label_about_flushot | majority | None | dev | {} | {} | {} | {} | {}"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

print("flu-risk | label_about_flushot | majority | None | test | {} | {} | {} | {} | {}"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))



