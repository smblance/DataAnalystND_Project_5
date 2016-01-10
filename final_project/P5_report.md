## Project Report for P5, Udacity NanoDegree.
### Investigating Enron Dataset with Machine Learning.
_Mark Ayzenshtadt, Jan 2016._

### 0. Project File Structure
All relevant files are contained in the __final_project__ directory of the repo.

Files provided by Udacity for the project:<br />__final\_project\_dataset.pkl__ - dataset to be used for the project.<br />__tester.py__ - script for testing classifier performance.<br />__enron61702insiderpay.pdf__ - financial data on the Enron employees.<br />__poi\_names.txt/poi\_email\_addresses.py__ - names and email addresses of the people involved in the fraud and their presence in the dataset.<br />__../tools/feature\_format.py__ - script for converting data from dictionary format to a python list.


Files made by me for the project:<br />__P5\_report.md / P5\_report.html__ - this report.<br />__poi\_id.py__ - the final version of the classifying script.<br />__poi\_id\_in\_process.py__ - intermediate (dirty) version of the classifying script.<br />__custom\_classifiers.py__ - contains the custom classifiers made for this project and used in poi\_id.py.<br />__enron\_to\_csv.py__ - exports project data to __final\_project\_data.csv__ for use in R.<br />__enron\_exploration.Rmd__ - some statistics and plots in R Markdown.<br />__log.txt__ - log of the parameters and performance metrics of the classifiers tested with tester.py.<br />__my\_classifier.pkl
, my\_dataset.pkl, my\_feature\_list.pkl__ - classifier, dataset and feature_list exported by poi\_id.py.<br />__references.txt__ - reference list.


### 1. Overview
__Enron__ was an American corporation that went bankrupt in 2001 because of a billon-dollar fraud, which involved many of its employees and contractors.<br />This project is based on the Enron dataset, which contains over 600,000 emails from some of the Enron's employees, that, along with the data on financial compensation to the employees, was made available after the trial.
Employees that participated in the fraud (according to [this USA Today article](http://usatoday30.usatoday.com/money/industries/energy/2005-12-28-enron-participants_x.htm)) are marked as __person of interest__ (__poi__).<br />There are 18 poi in the dataset.

The task is to build a model that will predict whether or not a person is a poi by using only financial and email information.
This task of binary classification is covered by variety of machine learning algorithms, like K Nearest Neighbors, Decision Trees or SVM.

The dataset that I'll be using in the project has 145 entries (people), 14 financial features and 6 features that cover email statistics.<br />All of the features are numeric (integers).<br />The labels that we want to predict - __poi__ - are boolean values.

Most of the features have missing values - for example `loan_advances` has only 3 non-NaN entries.<br />Using the provided __enron61702insiderpay.pdf__ document, I've made a conclusion that NaNs in the financial data actually represent zeros.<br />NaNs in email information mean that we don't have access to a person's emails, so I chose to replace NaN with a feature's median value.

I've started by examining the dataset in R and finding about 15 outliers.<br />One outlier - TOTAL - is an aggregate value for all points, and thus isn't a valid entry.<br />I removed it.<br />Other outliers seem plausible, and there was no evidence to remove them.


### 2. Feature selection and scaling
First, I added a feature `'part_of_incoming_from_poi' = 'from_poi_to_this_person'/'to_messages'.`<br />This feature shows what part of incoming messages to a person was from a poi, and I expect high level of this feature to mean that the person is also a poi.<br />Adding this feature didn't not change the F1-score, so I didn't add it to the final list.

I've used SelectKBest for feature selection.<br />These are the scores of the features:<br />bonus - 6.9<br />total\_payments - 3.19<br />long\_term\_incentive - 2.28<br />exercised\_stock\_options - 1.19<br />director\_fees - 1.15<br />other - 0.94<br />salary - 0.91<br />restricted\_stock - 0.82<br />deferral\_payments - 0.63<br />restricted\_stock\_deferred - 0.52<br />from\_messages - 0.18<br />to\_messages - 0.15<br />shared\_receipt\_with\_poi - 0.13<br />from\_this\_person\_to\_poi - 0.13<br />from\_poi\_to\_this\_person - 0.12<br />deferred\_income - 0.09<br />total\_stock\_value - 0.03<br />expenses - 0.02<br />part\_of\_incoming\_from\_poi - nan<br />loan\_advances - nan<br />

At first, I took 9 features with highest scores: bonus, total\_payments, long\_term\_incentive, exercised\_stock\_options, director\_fees, other, salary, restricted\_stock, deferral\_payments, and restricted\_stock\_deferred.

Then I experimented with adding or removing some of the features to this list and tried to improve the score using different feature combinations.<br />Mainly, I added/removed a feature and looked at the score, but sometimes the presence of some other feature changed the influence of the feature on the score, so this process was more of trial-and-error. <br />There aren't too many features so I'm fairly sure that the score cannot be improved with changing set of features for the model.<br />This is the list I ended up with: bonus, total\_payments, long\_term\_incentive, exercised\_stock\_options, other, restricted\_stock, deferral\_payments, and deferred\_income.<br />Adding or removing any features either reduces or does not change the F1-score.

I deployed scaling with sklearn's MaxAbsScaler, MinMaxScaler, StandardScaler, Normalizer, but using any of these with the modified k-NN algorithms that I ended up using reduced F1-score from 0.4-0.5 to 0.1-0.2, so scaling is not utilised.

### 3. Algorithm choice and performance
It would be logical to discuss evaluation metrics before discussing how I was choosing an algorithm.<br />The top-of-mind evaluation metric is __accuracy__ - correct fraction of predictions. But when we have an unbalanced dataset - in this case 12% to 88% (7 times less poi that non-poi) - even a trivial one-class classifier will have 88% accuracy.<br />Next natural step is to look at the __preicision__ and __recall__ metrics.<br />Precision shows what part of positive predictions (poi) was true, and recall shows what part of true predictions was predicted as true.<br />Most algorithms have trouble fitting to unbalanced data (understanding that the rare class is rare) and will thus have a lot of false-positive predictions. This will lead to naturally low values of precision.<br />On the other side, unbalanced data does not stop an algorithm from fitting to the rare class and thus recall values are often considerable.<br />So, I chose precision as the first target, and after acquiring 0.3 precision, I proceeded to tune the algorithm for higher __F1-score__, which is an aggregate measure of precision and recall.<br />To test different algorithms I used the provided tester.py script, with folds = 100.

I tried a wide variety of algorithms from __sklearn__ library - namely, LinearSVC/SVC, KNeighborsClassifier, RadiusNeighborsClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier, DecisionTreeClassifier and LogisticRegression.
I used scaling via Normalizer, MinMaxScaler or StandardScaler, and PCA and SelectKBest for feature selection, incorporating them into classifier via Pipeline.

After not being satisfied with the performance, I've made two strategies for dealing with unbalanced data: __splitter__ and __replicator__.<br />Both of them take a classifier at construction and work as classifiers themselves.

__Splitter:__<br />Inits with a classifier and parameters - `certainty`, `max_imbalance` and `clf_mode`.<br />When it fits to the data, it splits the frequent class (non-poi) into number of samples so that each sample has only `max_imbalance` more points than the rare class.<br />Then it constructs number of classifiers controlled by `clf_mode` (`clf_mode = 0` means 1 classifier is made for each sample from the split), and trains each on one part of the split and full rare class. This way class imbalance for each classifier can be manually controlled.<br />At prediction, each of the classifiers is predicting a value of a point and the final prediction is done by voting.<br />The minimal number of poi votes that needs be met so that the splitter predicts poi can be controlled by the `certainty` parameter.

__Replicator:__<br />Inits with the classifier and `dominant_class_prevalence` parameter.<br />At fitting, the classifier fits to all points of the frequent class, and points of the rare class replicated some number of times so that the frequent class has only `dominant_class_prevalence` more points than the rare class.<br />Prediction is simply done with the fitted classifier.

The only classifier using which I could consistently achieve precision greater than 0.3 (using splitter/replicator) was k-NN. Ridge classifier was close, with precision about 0.28.

When I focused on k-NN, I understood that I wanted to give more weight to the votes of the neighbors of the rare class. As it wasn't possible in the standard sklearn implementation, I've made the __weighted\_knn__ classifier.<br />It mirrors the functionality of sklearn version of k-NN as it has `n_neighbors` and `distance_weights` (`distance` in sklearn) parameters.<br />It also has `class_weights` parameter, a dictionary of weights of the classes, which is applied at the voting process. The default value is `balanced`, which makes class weights equal to inverse class frequencies (rare class = more weight).<br />Weighted\_knn is equal to standard k-NN if class weights are equal, and the scores proved to be identical.<br />The difference between replicator+knn and weighted\_knn is that in weighted\_knn, poi points occupy only one neighbor slot, and in replicator+knn one poi point act as more that one neighbor, as it is replicated. Replication puts more weight to the poi votes, as does class\_weight in weighted\_knn.

All 3 algorithms are present in the final __poi_id.py__ script.

### 4. Hyperparameter optimization
Most of the algorithms will not work out-of-the-box with every data.<br />To achieve desired performance, you need to change the internal parameters of the algorithm. This process is called hyperparameter optimization, and it's done to ensure that the model generalizes well to the data it was not trained on.

One of the useful methods is grid search, implemented in sklearn as GridSearchCV. It tries every possible combination of parameters from the supplied lists and through cross validation finds the combination of parameters that maximizes the chosen metric.<br />I tried using GridSearchCV with sklearn algorithms, but in such a broad search, it was too slow and I went to manually choosing parameters.<br />Also, I've duplicated the output to the _log.txt_ file to have a history of my search.

All three algorithms described in section 3 quite performed well. The scores that I managed to achieve are:

`splitter (max_imbalance = 2, certainty = 0.6, clf_class = KNeighborsClassifier, clf_mode = 0, n_neighbors = 3, weights = 'uniform')`<br />has __F1 = 0.40707__.

`replicator (dominant_class_prevalence = 2, clf_class = KNeighborsClassifier, n_neighbors = 12, weights = 'uniform')`<br />has __F1 = 0.38360__.

`weighted_knn (n_neighbors = 6, class_weights= {0:1, 1:5}, distance_weights = 'uniform')`<br />has __F1 = 0.39629__.

`weighted_knn (n_neighbors = 10, class_weights = {0:1, 1:3}, distance_weights = 'uniform')`<br />has __F1 = 0.50935__.

All four of these models are present in the final poi\_id.py, and the last one is used for classifying.

### 5. Cross-validation
Cross-validation is a method of ensuring that the model is generalizing well to the data that it was not trained on.<br />It is done by splitting the data into training and testing sets, training the algorithm on the training set and assessing the performance on the test set.<br />One classic mistake that can be done is testing the performance on the training set. This way you have no way of assesing if the data was overfit to the training set, and the performance will naturally be very high.<br />We want to know how the model will perform on the data that it wasn't trained on, and thus we should carefully look at what data we're testing the algorithm on.

Our dataset is imbalanced in respect to the distribution between classes, and we want our cross validation splits to have the same class distribution.<br />Sklearn's StratifiedShuffleSplit and StratifiedKFold are made to adress this issue, and I chose StratifiedShuffleSplit as in tester.py.

I tested the algorithm's performance with provided tester.py script, with folds = 100 at the search stage and folds = 1000 in the final tuning.

### 6. Evaluation metrics
Evaluation metrics were discussed at the beginning of section 3.


