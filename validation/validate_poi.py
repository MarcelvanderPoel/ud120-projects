#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import collections
from sklearn.metrics import classification_report

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


### it's all yours from here forward!

### split the data in 30% test data and 70% training data

training_features, testing_features, training_labels, testing_labels = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

### fit decision tree, predict and determine accuracy

clf = tree.DecisionTreeClassifier()
clf.fit(training_features, training_labels)
pred = clf.predict(testing_features)
accuracy = accuracy_score(testing_labels, pred)
print 'accuracy: ', accuracy
print 'total people in test set: ', len(testing_features)
print collections.Counter(testing_labels)
print len(training_features)
print len(training_labels)
print zip(pred, testing_labels)
print classification_report(testing_labels, pred, target_names=['0','1'])