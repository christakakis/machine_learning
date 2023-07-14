# Classification (Supervised Learning)
## Financial Risk Assessment and Data Analytics

This repository contains Python code for performing financial risk assessment and data analytics on a dataset of Greek companies. The goal is to develop a classification model that can identify companies that are at risk of bankruptcy.

## Part A: Classification Model Development

In this section, we will focus on creating the best possible classification model to identify companies that may go bankrupt. The dataset includes the following information for each company:

  1. Performance indicators (columns A to H)
  2. Dual activity indicators (columns I, J, K)
  3. Company state (1 for all good, 2 for bankruptcy)
  4. Year of the indicators

### Detailed Description

The procedure involves the following steps:

  1. Reading the data from an Excel file, normalizing it, and dividing it into training and test sets.
  2. Implementing and evaluating several supervised learning models using Python:
     - Linear Discriminant Analysis
     - Logistic Regression
     - Decision Trees
     - k-Nearest Neighbors
     - Naïve Bayes
     - Support Vector Machines
     - Neural Networks

The performance of each model will be evaluated based on the following constraints:

1. The model must have a success rate of at least 62% for identifying companies that will go bankrupt.
2. The model must have a success rate of at least 70% for identifying companies that will not go bankrupt.

The results of the experiments will be transferred to an Excel file, where each row will include the following values:

    Classifier Name | Training or Test Set | Number of Training Samples | Number of Non-healthy Companies in Training Set | TP | TN | FP | FN | Precision | Recall | F1 Score | Accuracy

Based on the results on the test data, we will determine if any of the models satisfy the performance constraints.

## Part B: Adjusted Training Set

In this section, we will repeat the experiment with an adjusted training set. If the distribution of healthy firms to bankrupt firms in the original training set is greater than 3 to 1, we will randomly select healthy firms to balance the ratio to 3 healthy firms for every bankrupt firm.

---

This repository was initially created to store my personal python codes but also be available to others trying to build or understand something similar.
The codes contained in this repo are made specifically for a Machine Learning and Natural Language Processing course of my MSc program, as part of a data analysis and financial risk assessment project

Please note that the data used in this repository belongs to their respective owners, and the repository's purpose is to showcase analytical skills and code implementation.
