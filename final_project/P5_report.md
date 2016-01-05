## Project report for P5, Udacity NanoDegree.
### Investigating Enron Dataset with Machine Learning.
_Mark Ayzenshtadt, Jan 2016._

### 0. Project File Structure
All relevant files are contained in the __final_project__ directory of the repo.

Files provided by Udacity for the project:<br />__final\_project\_dataset.pkl__ - dataset to be used for the project.<br />__tester.py__ - script for testing classifier performance.<br />__enron61702insiderpay.pdf__ - financial data on the Enron employees.<br />__poi\_names.txt/poi\_email\_addresses.py__ - names and email addresses of the people involved in the fraud and their presence in the dataset.<br />__../tools/feature\_format.py__ - script for converting data from dictionary format to a python list.


Files made by me for the project:<br />__P5\_report.md / P5\_report.html__ - this report.<br />__poi\_id.py__ - the final version of the classifying script.<br />__poi\_id\_in\_process.py__ - intermediate (dirty) version of the classifying script.<br />__custom\_classifiers.py__ - contains the custom classifiers made for this project and used in poi\_id.py.<br />__enron\_to\_csv.py__ - exports project data to __final\_project\_data.csv__ for use in R.<br />__enron\_exploration.Rmd__ - some statistics and plots in R Markdown.<br />__log.txt__ - log of the parameters and performance metrics of the classifiers tested with tester.py.<br />__my\_classifier.pkl, my\_dataset.pkl, my\_feature\_list.pkl__ - classifier, dataset and feature_list exported by poi\_id.py.<br />__references.txt__ - reference list.


### 1. Overview
__Enron__ was an American corporation that went bankrupt in 2001 because of a billon-dollar fraud, which involved many of its employees and contractors.<br />This project is based on the Enron dataset, which contains over 600,000 emails from some of the Enron's employees, that, along with the data on financial compensation to the employees, was made available after the trial.
Employees that participated in the fraud (according to [this USA Today article](http://usatoday30.usatoday.com/money/industries/energy/2005-12-28-enron-participants_x.htm)) are marked as __person of interest__ (__poi__).<br />There are 18 poi in the dataset.

The task is to build a model that will predict whether or not a person is a poi by using only financial and email information.
This task of binary classification is covered by variety of machine learning algorithms, like K Nearest Neighbors, Decision Trees or SVM.

The dataset that I'll be using in the project has 145 entries (people), 14 financial features and 6 features that cover email statistics.<br />All of the features are numeric (integers).<br />The labels that we want to predict - __poi__ - are boolean values.

Most of the features have missing values - for example `loan_advances` has only 3 non-NaN entries.<br />Using the provided __enron61702insiderpay.pdf__ document, I've made a conclusion that NaNs in the financial data actually represent zeros.<br />NaNs in email information mean that we don't have access to a person's emails, so I chose to replace NaN with a feature's median value.

I've started by examining the dataset in R and finding some outliers.<br />As the presence of outliers is poorly affecting the performance of most of the algorightms, I've removed about 2-3 extreme points for each feature.<br />Outliers for most of the features overlap, as top executives have very high financial parameters.

### 2. Feature selection
First, I added a feature `'part_of_incoming_from_poi' = 'from_poi_to_this_person'/'to_messages'.`<br />This feature shows what part of incoming messages to a person was from a poi, and I expect high level of this feature to mean that the person is also a poi.<br />After testing, this feature didn't make it to the final list.

I started with using some of the features that I've found relevant in the exploration and tried different algorithms.<br />With every algorithm I examined if scaling/PCA/SelectKBest improved the performance of the classifier.

Scaling/PCA/Kbest didn't prove useful with the algorithm that I was content with (k-NN), and I came up with the following iterative process:<br />First, I tuned the parameters of the chosen algorithm with all available features.<br />Then I removed one feature and noted if it improved performance. After doing this  with every feature, I had a list of rather good features.<br />Then I iteratively reduced the list with the same method until I had a list from which I couldn't remove any single feature withiout reducing performance.<br />In the end, I was left with 8 financial features and no email features.

### 3. Algorithm choice and performance
It would be logical to discuss evaluation metrics before discussing how I was choosing an algorithm.<br />The top-of-mind evaluation metric is __accuracy__ - correct fraction of predictions. But when we have an unbalanced dataset - in this case 12% to 88% (7 times less poi that non-poi) - even a trivial one-class classifier will have 88% accuracy.<br />Next natural step is to look at the __preicision__ and __recall__ metrics.<br />Precision shows what part of positive predictions (poi) was true, and recall shows what part of true predictions was predicted as true.<br />Most algorithms have trouble fitting to unbalanced data (understanding that the rare class is rare) and will thus have a lot of false-positive predictions. This will lead to naturally low values of precision.<br />On the other side, unbalanced data does not stop an algorithm from fitting to the rare class and thus recall values are often considerable.<br />So, I chose precision as the first target, and after acquiring 0.3 precision, I proceeded to tune the algorithm for higher __F1-score__, which is an aggregate measure of precision and recall.<br />To test different algorithms I used the provided tester.py script, with folds = 100.

I tried a wide variety of algorithms from __sklearn__ library - namely, LinearSVC/SVC, KNeighborsClassifier, RadiusNeighborsClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier, DecisionTreeClassifier and LogisticRegression.
I used scaling via Normalizer, MinMaxScaler or StandardScaler, and PCA and SelectKBest for feature selection, incorporating them into classifier via Pipeline.

After not being satisfied with the performance, I've made two strategies for dealing with unbalanced data: __splitter__ and __replicator__.<br />Both of them take a classifier at construction and work as classifiers themselves.

__Splitter:__<br />Inits with a classifier and parameters - `certainty`, `max_imbalance` and `clf_mode`.<br />When it fits to the data, it splits the frequent class (non-poi) into number of samples so that each sample has only `max_imbalance` more points than the rare class.<br />Then it constructs number of classifiers controlled by `clf_mode` (`clf_mode = 0` means 1 classifier is made for each sample from the split), and trains each on one part of the split and full rare class. This way class imbalance for each classifier can be manually controlled.<br />At prediction, each of the classifiers is predicting a value of a point and the final prediction is done by voting.<br />The minimal number of poi votes that needs be met so that the splitter predicts poi can be controlled by the `certainty` parameter.

__Replicator:__<br />Inits with the classifier and `dominant_class_prevalence` parameter.<br />At fitting, the classifier fits to all points of the frequent class, and points of the rare class replicated some number of times so that the frequent class has only `dominant_class_prevalence` more points than the rare class.<br />Prediction is simply done with the fitted classifier.

The only classifier using which I could consistently achieve precision greater than 0.3 (using splitter/replicator) was k-NN. Ridge classifier was close, with precision about 0.28.

When I focused on k-NN, I understood that I wanted to give more weight to the votes of the neighbors of the rare class. As it wasn't possible in the standard sklearn implementation, I've made the __weighted\_knn__ classifier.<br />It mirrors the functionality of sklearn version of k-NN as it has `n_neighbors` and `distance_weights` (`distance` in sklearn) parameters.<br />It also has `class_weights` parameter, a dictionary of weights of the classes, which is applied at the voting process. The default value is `balanced`, which makes class weights equal to inverse class frequencies (rare class = more weight).<br />Weighted\_knn is equal to standard k-NN if class weights are equal, and the scores proved to be identical.<br />The difference between replicator+knn and weighted\_knn is that in weighted\_knn, poi points occupy only one neighbor slot, and in replicator+knn one poi point act as more that one neighbor, as it is replicated. Replication puts more weight to the poi votes, as does class\_weight in weighted\_knn.

All three of the mentioned algorithms performed well and showed F1-score of __0.44-0.47__ and are present in the __poi_id.py__ script.<br />The best F1-score that I managed to achieve was __0.47__ with weighted\_knn classifier, `n_neighbors = 10`, `class_weights = {0:1, 1:3}` and uniform distance weights.

### 4. Hyperparameter optimization
Most of the algorithms will not work out-of-the-box with every data.<br />To achieve desired performance, you need to change the internal parameters of the algorithm. This process is called hyperparameter optimization, and it's done to ensure that the model generalizes well to the data it was not trained on.

One of the useful methods is grid search, implemented in sklearn as GridSearchCV. It tries every possible combination of parameters from the supplied lists and through cross validation finds the combination of parameters that maximizes the chosen metric.<br />I tried using GridSearchCV with sklearn algorithms, but in such a broad search, it was too slow and I went to manually choosing parameters.<br />Also, I've duplicated the output to the _log.txt_ file to have a history of my search.

### 5. Cross-validation
Cross-validation is a method of ensuring that the model is generalizing well to the data that it was not trained on.<br />It is done by splitting the data into training and testing sets, training the algorithm on the training set and assessing the performance on the test set.<br />One classic mistake that can be done is testing the performance on the training set. This way you have no way of assesing if the data was overfit to the training set, and the performance will naturally be very high.<br />We want to know how the model will perform on the data that it wasn't trained on, and thus we should carefully look at what data we're testing the algorithm on.

Our dataset is imbalanced in respect to the distribution between classes, and we want our cross validation splits to have the same class distribution.<br />Sklearn's StratifiedShuffleSplit and StratifiedKFold are made to adress this issue, and I chose StratifiedShuffleSplit as in tester.py.<br />I tested the algorithm's performance with provided tester.py script, with folds = 100 at the search stage and folds = 1000 in the final tuning.

### 6. Evaluation metrics
Evaluation metrics were discussed at the beginning of section 3.


