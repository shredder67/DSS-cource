# Desicion Support Systems - Part II

This repository consists of my code notes for DSS course, which contents are mostly about classic approaches in data exploration and analysis.

It includes:

1. **Association Rules Learning**
2. **K-means Clusterisation**
3. **Linear Regression**
4. **Logistic Regression**
5. **Naive Bayes Classifier**
6. **Decision Tree**

Each algorithm is located in coresponding folder with _algname_example.py_ file, which you can use as a baselines to test out implementations.

## 1. ARL - Association Rules Learning

### 1.1 Description

Association Rules Learning is a method of data mining where data is represented as transactions of the form {id: [list of items]}. This is a rule based algorithm, which means that output is list of pairs in the form A -> B, where A and B are some sort of independent values, pairs can be interpreted as IF(A) : THEN(B) statements.

### 1.2 Implementation details

///

## 2. K-means Clusterisation

### 2.1 Description

K-means clusterisation is one the most ancient techniques used to cluster data. It classic implemention main drawback is that it is fully dependent on hyperparamter **k** - amount of clusters data will be split into. It also very fragile to problems like _curse of dimensionality_ and _feature scale_ (which are relevant for most K-anything algorithms). Output of algorithm also easily overfits into local optimum of WCSS (within-cluster sum of squares), which can be compensated by running algorithm multiple times with different seeds and averaging results.

### 2.3 Implementation details

///

## 3. Linear Regression

///

## 4. Logistic Regression

///

## 5. Naive Bayes Classifier

///

## 6. Decision Tree
