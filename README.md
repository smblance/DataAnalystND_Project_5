# DataAnalystND_Project_5
Project 5 for Data Analyst NanoDegree at Udacity.<br />

In the project I've made a Machine Learning model for finding persons of interest in the Enron dataset.

All relevant files are contained in the __final_project__ directory of the repo.<br />Please refer to the __final\_project/P5\_report.html__ for the project report.

Files provided by Udacity for the project:<br />__final\_project\_dataset.pkl__ - dataset to be used for the project.<br />__tester.py__ - script for testing classifier performance.<br />__enron61702insiderpay.pdf__ - financial data on the Enron employees.<br />__poi\_names.txt/poi\_email\_addresses.py__ - names and email addresses of the people involved in the fraud and their presence in the dataset.<br />__../tools/feature\_format.py__ - script for converting data from dictionary format to a python list.


Files made by me for the project:<br />__P5\_report.md__ - this report.<br />__poi\_id.py__ - the final version of the classifying script.<br />__poi\_id\_in\_process.py__ - intermediate (dirty) version of the classifying script.<br />__custom\_classifiers.py__ - contains the custom classifiers made for this project and used in poi\_id.py.<br />__enron\_to\_csv.py__ - exports project data to __final\_project\_data.csv__ for use in R.<br />__enron\_exploration.Rmd__ - some statistics and plots in R Markdown.<br />__log.txt__ - log of the parameters and performance metrics of the classifiers tested with tester.py.<br />__my\_classifier.pkl, my\_dataset.pkl, my\_feature_list.pkl__ - classifier, dataset and feature_list exported by poi\_id.py.