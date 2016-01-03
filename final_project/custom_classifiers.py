from sklearn.svm import LinearSVC
import random
import numpy as np

def split_nonpoi(nonpoi_indices, clf_count):
	"""Splits nonpoi_indices into (almost) even parts"""
	split = []
	sample_size = len(nonpoi_indices)/clf_count
	remaining_list = nonpoi_indices
	random.shuffle(remaining_list)
	for n in range(clf_count-1):
		split.append(remaining_list[:sample_size])
		remaining_list = remaining_list[sample_size:]
	split.append(remaining_list)
	return split



class balancer():

	def __init__(self, certainty = 0.5, max_imbalance = 1, clf_mode = 0, clf_class = LinearSVC, **clf_params):
		self.certainty = certainty
		self.max_imbalance = max_imbalance
		self.clf_mode = clf_mode
		self.clf_class = clf_class
		self.clf_params = clf_params
		self.print_params = True

	def fit(self, features, labels):
		if self.print_params:
			print 'Balancer params:'
			print 'Certainty: %s'%self.certainty
			print 'max_imbalance: %s'%self.max_imbalance
			print 'clf_mode: %s'%self.clf_mode
			print 'clf_class: %s'%self.clf_class
			print 'clf_params: %s\n'%self.clf_params
			with open('log.txt','a+') as f:
				print >> f, 'Balancer params:'
				print >> f, 'certainty: %s'%self.certainty
				print >> f, 'max_imbalance: %s'%self.max_imbalance
				print >> f, 'clf_mode: %s'%self.clf_mode
				print >> f, 'clf_class: %s'%self.clf_class
				print >> f, 'clf_params: %s\n'%self.clf_params
			self.print_params = False

		poi_count = int(sum(labels))
		if self.clf_mode == 0:
			clf_count = int((len(labels)-poi_count)/(poi_count*self.max_imbalance))
			if clf_count == 0:
				clf_count = 1
		else:
			clf_count = self.clf_mode
		clf_list = []
		# print len(features), features
		# print len(labels), labels
		# print clf_count, poi_count
		for n in range(clf_count):
			new_clf = self.clf_class(**self.clf_params)
			clf_list.append(new_clf)
		# print clf_list
		nonpoi_indices = [n for n in range(len(labels)) if labels[n] == 0]
		poi_indices =    [n for n in range(len(labels)) if labels[n] == 1]
		sampled_nonpoi_indices = split_nonpoi(nonpoi_indices, clf_count)
		#print len(labels), poi_count, sampled_nonpoi_indices
		splits = [(poi_indices + nonpoi_sample) for nonpoi_sample in sampled_nonpoi_indices]

		for n in range(clf_count):
			new_features = [features[i] for i in splits[n]]
			new_labels   = [labels[i]   for i in splits[n]]
			clf_list[n].fit(new_features, new_labels)
		self.models = clf_list
		return self

	def predict(self, features):
		model_predictions = [model.predict(features) for model in self.models]
		feature_preds = [[pred[f] for pred in model_predictions] for f in range(len(features))]
		# print len(feature_preds)
		# print feature_preds
		pred = [(sum(prediction_list) > len(prediction_list)*self.certainty) \
				for prediction_list in feature_preds]
		
		# print 'pred', len(pred)
		# print 'features', len(features)
		# print features

		return pred

class replicator():

	def __init__(self, dominant_class_prevalence = 1, clf_class = LinearSVC, **clf_params):
		self.dominant_class_prevalence = dominant_class_prevalence
		self.clf_class = clf_class
		self.clf_params = clf_params
		self.clf = clf_class(**clf_params)
		self.print_params = True

	def fit(self, features, labels):
		if self.print_params:
			print 'Replicator params:'
			print 'dominant_class_prevalence: %s'%self.dominant_class_prevalence
			print 'clf_class: %s'%self.clf_class
			print 'clf_params: %s\n'%self.clf_params
			self.print_params = False
		poi_count = int(sum(labels))
		nonpoi_indices = [n for n in range(len(labels)) if labels[n] == 0]
		poi_indices =    [n for n in range(len(labels)) if labels[n] == 1]
		new_indices = nonpoi_indices + poi_indices*int((float(len(labels)-poi_count)/ poi_count / self.dominant_class_prevalence))
		#print poi_indices, int((float(len(labels)-poi_count)/ poi_count / self.dominant_class_prevalence))
		#new_indices = nonpoi_indices + list(np.random.choice(poi_indices, size = int((float(len(labels)-poi_count)/ poi_count / self.dominant_class_prevalence)), replace = True))
		new_features = [features[i] for i in new_indices]
		new_labels   = [labels[i]   for i in new_indices]
		self.clf.fit(new_features, new_labels)
		return self

	def predict(self, features):
		return self.clf.predict(features)


class weighted_knn():
	def __init__ (self, n_neighbors = 10, class_weights = 'balanced', distance_weights = 'uniform'):
		self.n_neighbors = n_neighbors
		self.class_weights = class_weights
		self.distance_weights = distance_weights

	def fit(self, features, labels):
		if self.class_weights == 'balanced':
			poi_count = sum(labels)
			nonpoi_count = len(labels) - poi_count
			self.class_weights = {0 : len(labels)/nonpoi_count, 1: len(labels)/poi_count}
		self.train_features = features
		self.train_labels = labels


	def predict(self, X):
		pred = [None]*len(X)
		for i in range(len(X)):
			distances = []
			for train_point in self.train_features:
				distances.append(sum([(train_point[n] - X[i][n])**2 for n in range(len(X[i]))]) ** 0.5)
			distances = sorted(list(enumerate(distances)), key = lambda x: x[1])
			neighbor_indices = [x[0] for x in distances[:self.n_neighbors]]
			neighbor_votes = [self.train_labels[n] for n in neighbor_indices]
			if self.distance_weights == 'uniform':
				pred[i] = sum(neighbor_votes) * self.class_weights[1] > \
						  (self.n_neighbors - sum(neighbor_votes)) * self.class_weights[0]
			elif self.distance_weights == 'distance':
				neighbor_weights = 1./np.array([distances[n][1] for n in neighbor_indices])
				pred[i] = sum([neighbor_weights[n] for n in range(len(neighbor_weights)) if neighbor_votes[n] == 1]) * self.class_weights[1] > \
						  sum([neighbor_weights[n] for n in range(len(neighbor_weights)) if neighbor_votes[n] == 0]) * self.class_weights[0]
		return pred



















