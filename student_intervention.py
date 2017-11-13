# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 22:35:29 2017
@author: Biswajith Gopinathan

"""
# Building a student Intervention Progam

"""1. Classification vs Regression
Your goal is to identify students who might need early intervention - which type of supervised machine learning problem is this, classification or regression? Why?
The type of problem is 'Classification' as the objective is to find out the students will pass or not."""

# Import the libraries 
import numpy as np
import pandas as pd

# Read Student Data
student_data = pd.read_csv("student-data.csv")
print ("Student Data read succefully !!")

# Display the below values about the datasets
n_students = len(student_data)
n_features = student_data.shape[1]-1
n_passed = len(student_data[student_data['passed']== 'yes'])
n_failed = len(student_data[student_data['passed']== 'no'])
grad_rate = 100.0* n_passed/n_students

# Display the details as we done so far
print ("Total number of students: {}".format(n_students))
print ("Total number of students who passed {}".format(n_passed))
print ("Total number of students who failed {}".format(n_failed))
print ("Number of features : {}".format(n_features))
print ("Graduation rate of the class {0:.2f}".format(grad_rate))

#Preparing the Data for medelling, trainig and testing
#Identify feature and target columns - Data contains non-numeric features. Machine learning algorithms expect numeric values to perform computations
# Extract feature (X) and target (y) columns

feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print ("Feature column(s):-\n{}".format(feature_cols))
print ("Target column: {}".format(target_col))

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print ("\nFeature values:-")
print (X_all.head()) # print the first 5 rows


# Preprocess feature columns by converting categorical columns to numerical columns 
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty
    
    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' =>                     'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX
X_all = preprocess_features(X_all)
print ("Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns)))
    
# Split data into training and test sets split the data (both features and corresponding labels) into training and test sets.
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

# First, decide how many training vs test samples we want
num_all = student_data.shape[0]  # same as len(student_data)
num_train = 300  # about 75% of the data
num_test = num_all - num_train

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X_all,y_all,train_size=300)

print ("Training set: {} samples".format(X_train.shape[0]))
print ("Test set: {} samples".format(X_test.shape[0]))

# Training and Evaluating Models
# Currently implementing 3 supervised learning models that are available in scikit learn
# 1. Decision Trees
# 2. Gaussian Naive Bayes
# 3. Support Vector Machines

# Train a model
# This is a generic method which used for fitting the selected model
import time
def train_classifier(clf, X_train, y_train):
    print ("Training {}...".format(clf.__class__.__name__))
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    trainingtime=end-start
    print ("Done!\nTraining time (secs): {:.3f}".format(trainingtime))
    return trainingtime

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
# Fit model to training data
DTC_trainingtime_300=train_classifier(clf, X_train, y_train)  # note: using entire training set here

# Predict on training set and compute F1 score
from sklearn.metrics import f1_score
def predict_labels(clf, features, target):
    print ("Predicting labels using {}...".format(clf.__class__.__name__))
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print ("Done!\nPrediction time (secs): {:.3f}".format(end - start))
    return f1_score(target.values, y_pred, pos_label='yes')

train_f1_score = predict_labels(clf, X_train, y_train)
print ("F1 score for training set: {}".format(train_f1_score))

# Predict on test data
print ("F1 score for test set: {}".format(predict_labels(clf, X_test, y_test)))

# Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test):
    print ("------------------------------------------")
    print ("Training set size: {}".format(len(X_train)))
    train_classifier(clf, X_train, y_train)
    print ("F1 score for training set: {}".format(predict_labels(clf, X_train, y_train)))
    print ("F1 score for test set: {}".format(predict_labels(clf, X_test, y_test)))
    
    train_predict(clf, X_train[0:100], y_train[0:100], X_test, y_test)
    train_predict(clf, X_train[0:200], y_train[0:200], X_test, y_test)
    train_predict(clf, X_train, y_train, X_test, y_test)
    
# Implement Gaussian Nave Baye's Theorem for train classifier
    from sklearn.naive_bayes import GaussianNB
    clfG=GaussianNB()
    train_classifier(clfG, X_train, y_train)
    
# Implement prediction using different training set sizes
    train_predict(clfG, X_train[0:100], y_train[0:100], X_test, y_test)
    train_predict(clfG, X_train[0:200], y_train[0:200], X_test, y_test)
    train_predict(clfG, X_train, y_train, X_test, y_test)
    
# Implement Support Vector Machine for train classifier
    from sklearn.svm import SVC
    clf=SVC()
    train_classifier(clf, X_train, y_train)
    
    