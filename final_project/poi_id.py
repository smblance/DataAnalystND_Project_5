#!/usr/bin/python

import sys
import pickle
import numpy as np
import random
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = [ \
	# 'salary', 
	'deferral_payments',
	'total_payments',
	# 'loan_advances',
	'bonus',
	# 'restricted_stock_deferred', 
	'deferred_income', 
	# 'total_stock_value', 
	# 'expenses', 
	'exercised_stock_options', 
	'other', 
	'long_term_incentive', 
	'restricted_stock', 
	# 'director_fees'
	]

email_features = [ \
	#'to_messages', 
	#'from_poi_to_this_person', 
	#'from_messages', 
	#'from_this_person_to_poi',
	#'shared_receipt_with_poi'
	]

features_list = ['poi'] + financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# impute data:
# finiancial: NaN -> 0
# email: NaN -> median
email_feat_values = {}
for feat in email_features:
	email_feat_values.update({feat : [person[feat] for person in data_dict.values() if person[feat]!= 'NaN']})
average_feat_values = {}
for feat in email_feat_values:
	average_feat_values.update({feat : int(np.median(email_feat_values[feat]))})
for name in data_dict:
	for feature in data_dict[name]:
		if data_dict[name][feature] == 'NaN':
			if feature in email_features:
				data_dict[name][feature] = average_feat_values[feature]
			else:
				data_dict[name][feature] = 0

# remove outliers
names_initial = data_dict.keys()
data_dict = {k:v for k,v in data_dict.iteritems() if v['salary'] < 6e5}
data_dict = {k:v for k,v in data_dict.iteritems() if v['deferral_payments'] < 2e6}
data_dict = {k:v for k,v in data_dict.iteritems() if v['total_payments'] < 9e7}
data_dict = {k:v for k,v in data_dict.iteritems() if v['bonus'] < 5e6}
data_dict = {k:v for k,v in data_dict.iteritems() if v['restricted_stock_deferred'] < 1e6}
data_dict = {k:v for k,v in data_dict.iteritems() if v['deferred_income'] > -2e6}
data_dict = {k:v for k,v in data_dict.iteritems() if v['total_stock_value'] < 2e7}
data_dict = {k:v for k,v in data_dict.iteritems() if v['expenses'] < 1.5e5}
data_dict = {k:v for k,v in data_dict.iteritems() if v['exercised_stock_options'] < 1e7}
data_dict = {k:v for k,v in data_dict.iteritems() if v['other'] < 3e6}
data_dict = {k:v for k,v in data_dict.iteritems() if v['long_term_incentive'] < 3e6}
data_dict = {k:v for k,v in data_dict.iteritems() if v['restricted_stock'] < 5e6}
data_dict = {k:v for k,v in data_dict.iteritems() if v['director_fees'] < 3e6}
data_dict = {k:v for k,v in data_dict.iteritems() if v['to_messages'] < 1e4}
data_dict = {k:v for k,v in data_dict.iteritems() if v['from_poi_to_this_person'] < 400}
data_dict = {k:v for k,v in data_dict.iteritems() if v['from_messages'] < 4000}
data_dict = {k:v for k,v in data_dict.iteritems() if v['from_this_person_to_poi'] < 300}
data_dict = {k:v for k,v in data_dict.iteritems() if v['shared_receipt_with_poi'] < 4000}

# print 'Names of the outlers removed:', [name for name in names_initial if name not in data_dict.keys()]

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict

# add feature 'part_of_incoming_from_poi' = 'from_poi_to_this_person'/'to_messages'
# as the feature doesn't improve the final score, it's commented out
# for name in my_dataset:
# 	if my_dataset[name]['from_poi_to_this_person'] != 0 and my_dataset[name]['to_messages'] != 0:
# 		my_dataset[name].update({'part_of_incoming_from_poi' : my_dataset[name]['from_poi_to_this_person']/my_dataset[name]['to_messages']})
# 	else:
# 		my_dataset[name].update({'part_of_incoming_from_poi' : 0})
# features_list.append('part_of_incoming_from_poi')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers. 

from sklearn.neighbors import KNeighborsClassifier
from custom_classifiers import *

# F1 = 0.44
splitter_knn_clf = splitter(max_imbalance = 2, 
					certainty = 0.6,
					clf_class = KNeighborsClassifier,
					clf_mode = 0,
					n_neighbors = 3,
					weights = 'uniform')

# F1 = 0.46
replicator_knn_clf = replicator(dominant_class_prevalence = 2,
						clf_class = KNeighborsClassifier,
						n_neighbors = 12,
						weights = 'uniform')

# F1 = 0.45
weighted_knn_clf_1 = weighted_knn(n_neighbors = 6, 
						class_weights= {0:1, 1:5}, 
						distance_weights = 'uniform')

# F1 = 0.47
weighted_knn_clf_2 = weighted_knn(n_neighbors = 10, 
						class_weights = {0:1, 1:3}, 
						distance_weights = 'uniform')


clf = weighted_knn_clf_2

# adding any of the following scalers reduces F1-score to 0.1-0.2,
# so no scaling is deployed
# from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, Normalizer
# from sklearn.pipeline import Pipeline
# scalers = [MaxAbsScaler(), MinMaxScaler(), StandardScaler(), Normalizer()]
# clf = Pipeline([('scaler',scalers[0]), ('clf',splitter_knn_clf)])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

from tester import test_classifier

folds = 1000
test_classifier(clf, my_dataset, features_list, folds = folds)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)