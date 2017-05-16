#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy
from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train = features_train[:len(features_train) / 100]
# labels_train = labels_train[:len(labels_train) / 100]

#########################################################
### your code goes here ###

C=10000

print "C: ", C
clf=SVC(kernel="rbf", C=C)



t0 = time()
clf.fit(features_train, labels_train)
print "training time: ", round(time()-t0, 3), "s"

t1 = time()
pred=clf.predict(features_test)
print "prediction time: ", round(time()-t1, 3), "s"

accuracy=accuracy_score(labels_test, pred)
print "accuracy: ", accuracy
#########################################################

print numpy.count_nonzero(pred)
