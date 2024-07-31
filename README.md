# Classification Challenge - Spam Detector

## Overview

This notebook demonstrates the process of creating and comparing two different machine learning models to detect spam. The workflow includes data retrieval, preprocessing, model training, and evaluation of model performance.

## Contents

1. **Data Retrieval**
2. **Data Preprocessing**
   - Splitting the data into training and testing sets
   - Scaling the features
3. **Model Training**
   - Logistic Regression
   - Random Forest Classifier
4. **Model Evaluation**

## Steps

### 1. Data Retrieval

The dataset is sourced from the [UCI Machine Learning Library](https://archive.ics.uci.edu/dataset/94/spambase) and can be accessed [here](https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv). The data is imported using Pandas.

### 2. Data Preprocessing

#### a. Splitting the Data

The data is split into training and testing sets using the `train_test_split` function from `sklearn.model_selection`.

#### b. Scaling the Features

The `StandardScaler` from `sklearn.preprocessing` is used to scale the feature data. Both `X_train` and `X_test` are scaled to ensure that the models are trained on normalized data.

### 3. Model Training

Two types of models are created, trained, and evaluated:

#### a. Logistic Regression

A Logistic Regression model is created, fitted to the training data, and predictions are made using the testing data. The accuracy score of the model is printed.

#### b. Random Forest Classifier

A Random Forest Classifier model is created, fitted to the training data, and predictions are made using the testing data. The accuracy score of the model is printed.

### 4. Model Evaluation

The performance of the Logistic Regression and Random Forest models is compared. The accuracy scores of both models are used to determine which model performs better. The results and observations are documented.

## Usage

To run the notebook, ensure that you have the necessary libraries installed:

```bash
pip install pandas scikit-learn
