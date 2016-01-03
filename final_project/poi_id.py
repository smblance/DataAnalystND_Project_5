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
	#'salary', 
	'deferral_payments',
	'total_payments',
	'bonus',
	#'restricted_stock_deferred', 
	'deferred_income', 
	#'total_stock_value', 
	#'expenses', 
	'exercised_stock_options', 
	'other', 
	'long_term_incentive', 
	'restricted_stock', 
	#'director_fees'
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
#replace NaN with 0
random.seed(42)

email_feat_values = {}
for feat in email_features:
	email_feat_values.update({feat : [person[feat] for person in data_dict.values() if person[feat]!= 'NaN']})
average_feat_values = {}
for feat in email_feat_values:
	average_feat_values.update({feat : int(np.median(email_feat_values[feat]))})
#print average_feat_values
for name in data_dict:
	for feature in data_dict[name]:
		if data_dict[name][feature] == 'NaN':
			if feature in email_features:
				data_dict[name][feature] = average_feat_values[feature]
			else:
				data_dict[name][feature] = 0

#remove outliers
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

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict

#add 'part_of_incoming_from_poi' = 'from_poi_to_this_person'/'to_messages'
#the feature doesn't improve the final score
# for name in my_dataset:
# 	if my_dataset[name]['from_poi_to_this_person'] != 0 and my_dataset[name]['to_messages'] != 0:
# 		my_dataset[name].update({'part_of_incoming_from_poi' : my_dataset[name]['from_poi_to_this_person']/my_dataset[name]['to_messages']})
# 	else:
# 		my_dataset[name].update({'part_of_incoming_from_poi' : 0})
#features_list.append('part_of_incoming_from_poi')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Imputer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from custom_classifiers import *

minmaxscaler = MinMaxScaler()
stdscaler = StandardScaler()

nb = GaussianNB()
etc = ExtraTreesClassifier(n_estimators = 50, min_samples_split = 1, max_features = None, class_weight = 'balanced')
rfc = RandomForestClassifier( \
	min_samples_split=1, max_features = None,
	max_depth = None,  
	class_weight = 'balanced', bootstrap = False)
gradboost = GradientBoostingClassifier()
ada = AdaBoostClassifier(learning_rate = .5, n_estimators = 100)
dcf = DecisionTreeClassifier()
sgd = SGDClassifier(class_weight = 'balanced', average = 1)
passiveaggr = PassiveAggressiveClassifier(class_weight = 'balanced')
ridge = RidgeClassifier(class_weight = 'balanced')
logistic = LogisticRegression(dual = False, class_weight = 'balanced')
knc = KNeighborsClassifier(n_neighbors = 10,
						algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute'][0],
						leaf_size = 30,
						weights = 'uniform',
						p = 2)
rnc = RadiusNeighborsClassifier()

imputer = Imputer(missing_values = -1, strategy = 'median')

normalizer = Normalizer()
kbest = SelectKBest(k = 17)
pca = PCA(n_components = 8)

linsvc = LinearSVC(dual = False, class_weight = 'balanced', C = 1)
svc = SVC(kernel = 'sigmoid', degree = 3, C = 10, class_weight = 'balanced')


# balancer = balancer(max_imbalance = 1, 
# 					certainty = 0.7,
# 					clf_class = RidgeClassifier,
# 					clf_mode = 0,
# 					class_weight = 'balanced',
# 					alpha = .01,
# 					normalize = True,
# 					solver = 'auto')

balancer = balancer(max_imbalance = 1, 
					certainty = 0.6,
					clf_class = KNeighborsClassifier,
					clf_mode = 0,
					n_neighbors = 10,
					algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute'][0],
					leaf_size = 30,
					weights = 'uniform',
					p = 2)

# replicator = replicator(dominant_class_prevalence = 1,
# 						clf_class = RidgeClassifier,
# 						class_weight = 'balanced',
# 						alpha = .01,
# 						normalize = True,
# 						solver = 'auto')

replicator = replicator(dominant_class_prevalence = 2,
						clf_class = KNeighborsClassifier,
						n_neighbors = 10,
						algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute'][0],
						leaf_size = 30,
						weights = 'uniform',
						p = 2)

pipe = Pipeline([ \
	#('scaler', normalizer),
	#('kbest', kbest), 
	#('pca', pca), 
	('clf', replicator)])

#clf = weighted_knn(n_neighbors = 6, class_weights={0:1, 1:2}, distance_weights = 'distance')
#clf1 = weighted_knn(n_neighbors = 5, class_weights={0:1, 1:2}, distance_weights = 'uniform')
clf = replicator


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
#print sorted(zip(clf.named_steps['kbest'].scores_, features_list[1:]), reverse = True)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)