# Desicion Support Systems - Part II

This repository consists of my code notes for DSS course, which contents are mostly about classic approaches in data exploration and analysis.

It includes:

1. **Association Rules Learning**
2. **K-means Clusterisation**
3. **Linear Regression**
4. **Logistic Regression**
5. **Naive Bayes Classifier**
6. **Decision Tree**

Each algorithm is located in coresponding folder with _algname_example.py_ file, which you can use as a baseline to test out implementations.

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

### 3.1 Description

Linear Regression is a supervised machine learning algorithm where the predicted output is continuous and has a constant slope. It’s used to predict values within a continuous range, (e.g. sales, price) rather than trying to classify them into categories (e.g. cat, dog). There are two main types:

- Simple Regression
- Multivariable regression

Simple regression is a special case of Multivariable regression. Model is generally formulted like

<img src="https://render.githubusercontent.com/render/math?math=y=w^Tx\%2Bb">

Where **y** is the predicted value, **w** is the weights verctor, **x** is the feature vector and **b** is a bias. Bias can be interpreted as a weight multiplied by constant 1. In a simple case where w and x are scalars, output of classification is a 2D line in XY coordinate system.

### 3.2 Implementation details

///

## 4. Logistic Regression

### 4.1 Description

///

### 4.2 Implementation details

///

## 5. Naive Bayes Classifier

### 5.1 Description

///

### 5.2 Implementation details

///

## 6. Decision Tree

### 6.1 Description

///

### 6.2 Implementation details

///
