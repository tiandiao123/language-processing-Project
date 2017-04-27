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
from process_data import handle_flu_json,handle_flu_risk_perception,handle_flu_vaccine_new
from process_data import handle_health_json,handle_trust_in_gov,handle_vaccine_sentiment,handle_zika_conspiracy
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


f=open("majority_classifer.txt","w+")
f.write("Dataset | Task | Model | FeatureSet | EvaluationSet | Accuracy | Precision | Recall | F1 Score | AUC\n")

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

f.write("flu | flu_relevant | majority | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

f.write("flu | flu_relevant | majority | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))


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

txt_about_flu=np.array(txt_about_flu)
prediction_labels_about_flu_likelihood=np.array(prediction_labels_about_flu_likelihood)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt_about_flu,prediction_labels_about_flu_likelihood)

f.write("flu-risk | label_about_flu_likelihood | majority | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

f.write("flu-risk | label_about_flu_likelihood | majority | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))

#case2:
txt_about_flu=[]
prediction_labels_about_flu=[]
for i in range(len(label_about_flu)):
    if label_about_flu[i]=='yes':
        prediction_labels_about_flu.append(1)
        txt_about_flu.append(train_data[i])
    elif label_about_flu[i]=='no':
        prediction_labels_about_flu.append(0)
        txt_about_flu.append(train_data[i])

txt_about_flu=np.array(txt_about_flu)
prediction_labels_about_flu=np.array(prediction_labels_about_flu)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt_about_flu,prediction_labels_about_flu)

f.write("flu-risk | label_about_flu | majority | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

f.write("flu-risk | label_about_flu | majority | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))

#case3:
txt_about_flu=[]
prediction_labels_about_flushot=[]
for i in range(len(label_about_fluShot)):
    if label_about_fluShot[i]=='yes':
        prediction_labels_about_flushot.append(1)
        txt_about_flu.append(train_data[i])
    elif label_about_fluShot[i]=='no':
        prediction_labels_about_flushot.append(0)
        txt_about_flu.append(train_data[i])

txt_about_flu=np.array(txt_about_flu)
prediction_labels_about_flushot=np.array(prediction_labels_about_flushot)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt_about_flu,prediction_labels_about_flushot)

f.write("flu-risk | label_about_flushot | majority | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

f.write("flu-risk | label_about_flushot | majority | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))



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

txt=np.array(txt)
prediction_labels_intend=np.array(prediction_labels_intend)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt,prediction_labels_intend)
f.write("flu-vaccine | label_flu_vaccine_intent_to_receive | majority | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

f.write("flu-vaccine | label_flu_vaccine_intent_to_receive | majority | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))



txt=[]
prediction_labels_flu_vaccine_relevant=[]
for i in range(len(label_flu_vaccine_relevant)):
    if label_flu_vaccine_relevant[i]=='yes':
        prediction_labels_flu_vaccine_relevant.append(1)
        txt.append(data[i])
    elif label_flu_vaccine_relevant[i]=='no':
        prediction_labels_flu_vaccine_relevant.append(0)
        txt.append(data[i])

txt=np.array(txt)
prediction_labels_flu_vaccine_relevant=np.array(prediction_labels_flu_vaccine_relevant)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt,prediction_labels_flu_vaccine_relevant)

f.write("flu-vaccine | flu_vaccine_relevant | majority | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

f.write("flu-vaccine | flu_vaccine_relevant | majority | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))


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

txt=np.array(txt)
prediction_labels=np.array(prediction_labels)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt,prediction_labels)

f.write("health.json | health-sick | majority | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

f.write("health.json | health-sick | majority | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))


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

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt,prediction_labels)

f.write("trust-in-government | label_trust_gov | majority | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

f.write("trust-in-government | label_trust_gov | majority | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))


txt=[]
prediction_labels=[]
for i in range(len(label_about_gov)):
    if label_about_gov[i]=='yes':
        prediction_labels.append(1)
        txt.append(data[i])
    elif label_about_gov[i]=='no':
        prediction_labels.append(0)
        txt.append(data[i])

txt=np.array(txt)
prediction_labels=np.array(prediction_labels)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt,prediction_labels)

f.write("trust-in-government | label_about_gov | majority | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

f.write("trust-in-government | label_about_gov | majority | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))


txt=[]
prediction_labels=[]
for i in range(len(label_about_vaccine)):
    if label_about_vaccine[i]=='yes':
        prediction_labels.append(1)
        txt.append(data[i])
    elif label_about_vaccine[i]=='no':
        prediction_labels.append(0)
        txt.append(data[i])

txt=np.array(txt)
prediction_labels=np.array(prediction_labels)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt,prediction_labels)

f.write("trust-in-government | label_about_vaccine | majority | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

f.write("trust-in-government | label_about_vaccine | majority | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))


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

txt=np.array(txt)
prediction_labels=np.array(prediction_labels)

dev_acc,dev_precision,dev_recall,dev_F1,dev_auc,test_accracy,Precision,Recall,F_1,auc=\
handle_two_classifications(txt,prediction_labels)

f.write("vaccine_sentiment | yes-no | majority | None | dev | {} | {} | {} | {} | {}\n"\
    .format(str(np.mean(dev_acc)),str(np.mean(dev_precision)),str(np.mean(dev_recall)),str(np.mean(dev_F1)),\
        str(np.mean(dev_auc))))

f.write("vaccine_sentiment | yes-no | majority | None | test | {} | {} | {} | {} | {}\n"\
    .format(str(test_accracy),str(Precision),str(Recall),str(F_1),str(auc)))

pprint("please check majority_classifier.txt")



