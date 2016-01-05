#!/usr/bin/python

import sys
import pickle
import csv
sys.path.append("../tools/")

financial_features = \
	['salary', 'deferral_payments', 'total_payments', 'loan_advances',
 	'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
 	'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
 	'restricted_stock', 'director_fees']

email_features = \
	['to_messages', 'email_address', 'from_poi_to_this_person', 
	'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

features_list = ['poi'] + financial_features + email_features
print features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# for feature in features_list:
# 	print feature
# 	print 'Number of non-NaN values:', len([x[feature] for x in data_dict.values() if x[feature] != 'NaN'])
# 	feature_values = set([x[feature] for x in data_dict.values()])
# 	print feature_values
# 	numerical_values = [x for x in feature_values if type(x)==int]
# 	if len(numerical_values) > 0:
# 		print 'Range:', min(numerical_values), max(numerical_values)
# 	print '----------------\n'

with open('final_project_data.csv','rb+') as csvfile:
	fieldnames = ['name'] + features_list
	writer = csv.DictWriter(csvfile, fieldnames)
	writer.writeheader()
	for name in data_dict:
		person = data_dict[name]
		person_and_name = person
		person_and_name.update({'name' : name})
		writer.writerow(person)