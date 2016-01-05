from sklearn.svm import LinearSVC
import random
import numpy as np

def split_nonpoi(nonpoi_indices, clf_count, random_seed, random_split = True):
	"""Split nonpoi_indices into (almost) even parts"""
	split = []
	sample_size = len(nonpoi_indices)/clf_count
	remaining_list = nonpoi_indices
	if random_split:
		random.seed(random_seed)
		random.shuffle(remaining_list)
	for n in range(clf_count-1):
		split.append(remaining_list[:sample_size])
		remaining_list = remaining_list[sample_size:]
	split.append(remaining_list)
	return split



class splitter():
	"""Binary classifier that uses the provided sklearn classifier 
	and splits the frequent class to achieve more balanced classes.
	Multiple instances of the internal classifier is used for training on the splits,
	and the prediction is done by voting.
	The rare class should be 1, frequent - 0.

	Parameters:
	----------
	certainty: float, 0...1, default = 0.5
		Controls how many of the mini-classifiers need to predict 1 
		so that the splitter predicts 1.

	max_imbalance: float, default = 1
		Controls the imbalance of the data that is feeded to each classifier

	clf_mode: int, default = 0
		Controls how many of the mini-classifiers are made.
		0 means that one classifier is made for one part of the split.

	random_seed: default = 42
		Seed used by the randomizer

	clf_class: sklearn classifier, default = LinearSVC
		The classifier that is used for fitting on part of the data.

	**clf_params: default = None
		Parameters passed to clf_class at construction.
	"""


	def __init__(self, certainty = 0.5, max_imbalance = 1, clf_mode = 0, random_seed = 42, clf_class = LinearSVC, **clf_params):
		self.certainty = certainty
		self.max_imbalance = max_imbalance
		self.clf_mode = clf_mode
		self.random_seed = random_seed
		self.clf_class = clf_class
		self.clf_params = clf_params
		self.print_params = True

	def fit(self, features, labels):
		"""Fit the algorithm to features and labels"""

		# print the params of the algorithm at first call to fit  
		if self.print_params:
			print 'Splitter params:'
			print 'Certainty: %s'%self.certainty
			print 'max_imbalance: %s'%self.max_imbalance
			print 'clf_mode: %s'%self.clf_mode
			print 'clf_class: %s'%self.clf_class
			print 'clf_params: %s\n'%self.clf_params
			with open('log.txt','a+') as f:
				print >> f, 'SplitterS params:'
				print >> f, 'certainty: %s'%self.certainty
				print >> f, 'max_imbalance: %s'%self.max_imbalance
				print >> f, 'clf_mode: %s'%self.clf_mode
				print >> f, 'clf_class: %s'%self.clf_class
				print >> f, 'clf_params: %s\n'%self.clf_params
			self.print_params = False

		# compute number of classifiers
		poi_count = int(sum(labels))
		if self.clf_mode == 0:
			clf_count = int((len(labels)-poi_count)/(poi_count*self.max_imbalance))
			if clf_count == 0:
				clf_count = 1
		else:
			clf_count = self.clf_mode
		
		# construct mini-classifiers
		clf_list = []
		for n in range(clf_count):
			new_clf = self.clf_class(**self.clf_params)
			clf_list.append(new_clf)

		# split the data
		nonpoi_indices = [n for n in range(len(labels)) if labels[n] == 0]
		poi_indices =    [n for n in range(len(labels)) if labels[n] == 1]
		sampled_nonpoi_indices = split_nonpoi(nonpoi_indices, clf_count, self.random_seed)
		splits = [(poi_indices + nonpoi_sample) for nonpoi_sample in sampled_nonpoi_indices]

		# fit mini-classifier
		for n in range(clf_count):
			new_features = [features[i] for i in splits[n]]
			new_labels   = [labels[i]   for i in splits[n]]
			clf_list[n].fit(new_features, new_labels)
		self.models = clf_list

		return self

	def predict(self, features):
		"""Predict the labels of the features using self.models classifier list and voting"""
		# predictons for each model
		model_predictions = [model.predict(features) for model in self.models]
		# predictions for each feature
		feature_preds = [[pred[f] for pred in model_predictions] for f in range(len(features))]

		# predict 1 if at least self.certainty of models predict 1
		pred = [(sum(prediction_list) > len(prediction_list)*self.certainty) \
				for prediction_list in feature_preds]

		return pred

class replicator():
	"""Binary classifier that uses the provided sklearn classifier 
	and replicates points of the rare class to achieve more balanced classes.
	The rare class should be 1, frequent - 0.

	Parameters:
	----------
	dominant_class_prevalence: float, default = 1
		Controls the imbalance of the data after replicating.

	randomize_replication: boolean, default = False

	random_seed: default = 42
		Seed used by the randomizer

	clf_class: sklearn classifier, default = LinearSVC
		The classifier that is used for fitting and prediction.

	**clf_params: default = None
		Parameters passed to clf_class at construction.
	"""

	def __init__(self, dominant_class_prevalence = 1, randomize_replication = False, random_seed = 42, clf_class = LinearSVC, **clf_params):
		self.dominant_class_prevalence = dominant_class_prevalence
		self.randomize_replication = randomize_replication
		self.random_seed = random_seed
		self.clf_class = clf_class
		self.clf_params = clf_params
		self.clf = clf_class(**clf_params)
		self.random_seed = random_seed
		self.print_params = True

	def fit(self, features, labels):
		"""Fit the algorithm to features and labels"""

		# print the params of the algorithm at first call to fit  
		if self.print_params:
			print 'Replicator params:'
			print 'dominant_class_prevalence: %s'%self.dominant_class_prevalence
			print 'clf_class: %s'%self.clf_class
			print 'clf_params: %s\n'%self.clf_params
			with open('log.txt','a+') as f:
				print >> f, 'Replicator params:'
				print >> f, 'dominant_class_prevalence: %s'%self.dominant_class_prevalence
				print >> f, 'clf_class: %s'%self.clf_class
				print >> f, 'clf_params: %s\n'%self.clf_params
			self.print_params = False

		# compute poi and non-poi indices
		poi_count = int(sum(labels))
		nonpoi_indices = [n for n in range(len(labels)) if labels[n] == 0]
		poi_indices =    [n for n in range(len(labels)) if labels[n] == 1]

		# replicate, randomly or not
		if not self.randomize_replication:
			new_indices = nonpoi_indices + poi_indices*int((float(len(labels)-poi_count)/ poi_count / self.dominant_class_prevalence))
		else:
			random.seed(self.random_seed)
			new_indices = nonpoi_indices + list(np.random.choice(poi_indices, size = int((float(len(labels)-poi_count)/ poi_count / self.dominant_class_prevalence)), replace = True))

		#fit the internal classifier
		new_features = [features[i] for i in new_indices]
		new_labels   = [labels[i]   for i in new_indices]
		self.clf.fit(new_features, new_labels)

		return self

	def predict(self, features):
		"""Predict labels of the features using internal classifier"""
		return self.clf.predict(features)


class weighted_knn():
	"""Binary classifier that uses K Nearest Neighbors algorith and can use different class weights.
	Mirrors in functionality sklearn's KNearestNeighbors.
	The rare class should be 1, frequent - 0.

	Parameters:
	----------
	n_neighbors: float, default = 10
		Number of neighbors to use for prediction.

	class_weights: 'balanced', 'uniform' or dict, default = 'balanced'
		Class weights used in prediction.
		'balanced': inversed frequencies.
		'uniform': equal weights.

	distance_weights: 'uniform' or 'distance', default = 'uniform'
		Distance weights used in prediction.
		'uniform': distance does not affect prediction
		'distance': in addition to class_weights, points have weights 
			equal to inverse distance to the predicted point.
	"""

	def __init__ (self, n_neighbors = 10, class_weights = 'balanced', distance_weights = 'uniform'):
		self.n_neighbors = n_neighbors
		self.class_weights = class_weights
		self.distance_weights = distance_weights
		self.print_params = True

	def fit(self, features, labels):
		"""Fit the algorithm to features and labels"""

		# print the params of the algorithm at first call to fit  
		if self.print_params:
			print 'weighted_knn params:'
			print 'n_neighbors: %s'%self.n_neighbors
			print 'class_weights: %s'%self.class_weights
			print 'distance_weights: %s\n'%self.distance_weights
			with open('log.txt','a+') as f:
				print >> f, 'weighted_knn params:'
				print >> f, 'n_neighbors: %s'%self.n_neighbors
				print >> f, 'class_weights: %s'%self.class_weights
				print >> f, 'distance_weights: %s\n'%self.distance_weights
			self.print_params = False

		# compute class weights
		if self.class_weights == 'balanced':
			poi_count = sum(labels)
			nonpoi_count = len(labels) - poi_count
			self.class_weights = {0 : len(labels)/nonpoi_count, 1: len(labels)/poi_count}
		elif self.class_weights == 'uniform':
			self.class_weights = {0: 1, 1: 1}

		# remember features and labels for prediction
		self.train_features = features
		self.train_labels = labels

		return self

	def predict(self, X):
		"""Predict labels of the features using features and labels passed to fit"""

		pred = [None]*len(X)

		for i in range(len(X)):

			# compute distances
			distances = []
			for train_point in self.train_features:
				distances.append(sum([(train_point[n] - X[i][n])**2 for n in range(len(X[i]))]) ** 0.5)
			distances = sorted(list(enumerate(distances)), key = lambda x: x[1])

			# comupte which points are neighbors
			neighbor_indices = [x[0] for x in distances[:self.n_neighbors]]

			# compute the unweighted votes of neighbors
			neighbor_votes = [self.train_labels[n] for n in neighbor_indices]

			# compute predictions
			if self.distance_weights == 'uniform':
				pred[i] = sum(neighbor_votes) * self.class_weights[1] > \
						  (self.n_neighbors - sum(neighbor_votes)) * self.class_weights[0]
			elif self.distance_weights == 'distance':
				# in this case weights are inverse distances
				neighbor_weights = 1./np.array([distances[n][1] for n in neighbor_indices])
				pred[i] = sum([neighbor_weights[n] for n in range(len(neighbor_weights)) if neighbor_votes[n] == 1]) * self.class_weights[1] > \
						  sum([neighbor_weights[n] for n in range(len(neighbor_weights)) if neighbor_votes[n] == 0]) * self.class_weights[0]

		return pred



















