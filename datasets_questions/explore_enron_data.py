#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import sys
import pickle
import numpy

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print enron_data["SKILLING JEFFREY K"]["total_payments"]
print enron_data["LAY KENNETH L"]["total_payments"]
print enron_data["FASTOW ANDREW S"]

d = dict((k, v) for k, v in enron_data.items() if v["total_payments"] == "NaN")

print len(d)

feature_list = ["total_payments", "poi"]
data_array = featureFormat(enron_data, feature_list, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=True, sort_keys = False )
print data_array


