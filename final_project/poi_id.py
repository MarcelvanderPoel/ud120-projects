#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import operator
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from feature_format import featureFormat, targetFeatureSplit

def compute_poi_communication_index(from_this_person_to_poi, from_poi_to_this_person, from_messages, to_messages):

    """ returns a number between 0 and 1 that represents the amount of communication between a person and a poi
        by multiplying the percentage of mail send to and received from a poi compared to the total mail send and
        received by a person.
   """

    if (from_this_person_to_poi == 0) or (from_poi_to_this_person == 0):
        from_poi_perc = 0.
    else:
        from_poi_perc = float(from_this_person_to_poi) / float(from_messages)

    if (from_messages == 0) or (to_messages == 0):
        to_poi_perc = 0.
    else:
        to_poi_perc = float(from_poi_to_this_person) / float(to_messages)

    from_to_add = from_poi_perc + to_poi_perc
    from_to_mlt = from_poi_perc * to_poi_perc

    return from_poi_perc, to_poi_perc, from_to_add, from_to_mlt


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees',
              'exercised_stock_options', 'expenses', 'from_messages',
              'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances',
              'long_term_incentive', 'other', 'restricted_stock', 'restricted_stock_deferred',
              'salary', 'shared_receipt_with_poi', 'to_messages', 'total_payments', 'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### transfer data_dict to a dataframe for easy printing and getting info on shape and counts

df = pd.DataFrame(data_dict)
df = df.T
df = df.applymap(lambda x: pd.np.nan if x=='NaN' else x)

print 'Shape of dataframe           : ', df.shape
print 'POI values and count         : ', df['poi'].value_counts()
print 'Non-Nan values per feature   : ', df.count()

### Task 2: Remove outliers

for key in ('THE TRAVEL AGENCY IN THE PARK','TOTAL', 'LOCKHART EUGENE E'): data_dict.pop(key)

### Task 3: Create new feature(s)

# Perform feature selection
data = featureFormat(data_dict, features_list)
poi, features = targetFeatureSplit(data)

# Create additional features for each data point and add them to data_dict and features
added_features = []

keys = data_dict.keys()
ft_count = 0
for key in keys:

    ft = features[ft_count]

    f_poi_p, t_poi_p, f_t_add, f_t_mult = compute_poi_communication_index(ft[9], ft[8], ft[7], ft[17])

    added_feature = [f_poi_p, t_poi_p, f_t_add, f_t_mult]
    added_features.append(np.array(added_feature))
    data_dict[key]['from_poi_perc'] = f_poi_p
    data_dict[key]['to_poi_perc'] = t_poi_p
    data_dict[key]['from_to_add'] = f_t_add
    data_dict[key]['from_to_mult'] = f_t_mult

    ft_count+=1

add_features = np.array(added_features)

features = np.concatenate((features,add_features),axis=1)

# Rescale features
scaler = MinMaxScaler()
rescaled_features = scaler.fit_transform(features)

# Select 10 best features
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(features, poi)
is_best_features = selector.get_support()

#remove 'poi' from the predictor list and print sorted feature_scores
predictors = features_list + ['from_poi_perc', 'to_poi_perc', 'from_to_add', 'from_to_mult']
predictors.pop(0)

feature_scores = zip (predictors, selector.scores_)
feature_scores.sort(key = operator.itemgetter(1), reverse = True)
print "Feature scores "
for feature_score in feature_scores:
    print feature_score

### Store to my_dataset for easy export below.
my_dataset = data_dict

keys = data_dict.keys()
for key in keys:

    my_dataset[key].pop('email_address')
    for counter in range(23):

        if is_best_features[counter] == False:
            my_dataset[key].pop(predictors[counter])


### Extract features and labels from dataset for local testing

features_list = ['poi']
for counter in range(23):
    if is_best_features[counter] == True:
        features_list.append(predictors[counter])

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Rescale features after reloading best features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)