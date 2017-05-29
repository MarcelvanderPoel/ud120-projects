#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### my code goes here ###

clf_2=tree.DecisionTreeClassifier(min_samples_split=2)
clf_2.fit(features_train, labels_train)
pred_2=clf_2.predict(features_test)

acc_min_samples_split_2=accuracy_score(labels_test, pred_2)
print "accuracy 2: ", acc_min_samples_split_2

clf_50=tree.DecisionTreeClassifier(min_samples_split=50)
clf_50.fit(features_train, labels_train)
pred_50=clf_50.predict(features_test)

acc_min_samples_split_50=accuracy_score(labels_test, pred_50)
print "accuracy 50: ", acc_min_samples_split_50


#########################################################


