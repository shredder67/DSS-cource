# Deсision Support Systems - Part II

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

**Apriori algorithm** is implemented for mining this rules from dataset. Algorithm mines rules based on user-defined constraines, that is *minimum support* and *minimum confidence* parameters. 

## 2. K-means Clusterisation

### 2.1 Description

K-means clusterisation is one the most ancient techniques used to cluster data. It classic implemention main drawback is that it is fully dependent on hyperparamter **k** - amount of clusters data will be split into. It also very fragile to problems like _curse of dimensionality_ and _feature scale_ (which are relevant for most K-anything algorithms). Output of algorithm also easily overfits into local optimum of WCSS (within-cluster sum of squares), which can be compensated by running algorithm multiple times with different seeds and averaging results.

## 3. Linear Regression

### 3.1 Description

Linear Regression is a supervised machine learning algorithm where the predicted output is continuous and has a constant slope. It’s used to predict values within a continuous range, (e.g. sales, price) rather than trying to classify them into categorie.

Simple regression is generally formulted like

$$
y = w^Tx + \varepsilon
$$

Where **y** is the predicted value, **w** is the weights verctor, **x** is the feature vector, **b** is a bias $\varepsilon$ is a stohastic error component. 

For measuring how good $w$ are, MSE-Loss is used:

$$
MSE = \frac{1}{N} \sum_{i=1}^N (y_i - w^Tx_i)^2
$$

Bias can be interpreted as a weight multiplied by constant 1. In a simple case where w and x are scalars, output of classification is a 2D line in XY coordinate system.

Implementation uses straitforward solution with following formula:
$$
w = (X^TX)^{-1}X^Ty
$$
No default feature normalization or regulization component for robustness of solution is provided.

## 4. Logistic Regression

### 4.1 Description

Logistic Regression is a model used mostly for binary classification problem. Predicted linear output of $w^Tx$ is treated as log-probabilities of positive label $ \log{\frac{p}{1 - p}} $.

To get actual probabilities from this, *sigmoid function* is used. So, the model consists of two parts:
$$
z(x) = w^Tx \\
\sigma (x) = \frac {1} {1 + e^{-z(x)}}
$$

The target function used for this problem is a special case of Cross Entropy Loss for binary classification, which is frequently just called *Logloss*:
$$
L(y, w, X) = \frac{1}{N} \sum^N_{i=1}[y_i\log{\sigma(x_i) + (1 - y_i)\log{\sigma(-x_i)}}]
$$

This problem doesn't have an analytical solution, so the standard *Gradient Descent* is used as optimization technique.

$$
\nabla_wL = x(\sigma(x) - y) \\
w_{i + 1} = w_i - \alpha\nabla_wL
$$

## 5. Naive Bayes Classifier

### 5.1 Description

Bayes Classifier is a probabalistic model, that learns probability distribution of features from training dataset and uses those to predict label based on input data. It does that using the following equiation:

$$
p(y | x) = \frac {p(x | y) p(y)} {p(x)}
$$

Deminator of this equation can be trivially expressed as $\sum_{y \in Y}p(x | y)$ over all possible labels. $p(y)$ can be calculated easily as apriory knowledge of labels using simple frequency from training data.

Naive Bayes Classifier makes an assumption about $x$, that all its components (which represent features) are independent, which makes possible to express $ p(x | y) $ as $ \prod^n_{i = 1}p(x_i | y)$. This really doesn't work in real world, since features usually have some sort of internal relation.

Simplest version of algorithm uses frequencies of categorial features and assumes normal distribution of any continuous feature by learning its $\mu$ and $\sigma$.
## 6. Decision Tree

### 6.1 Description

Decision tree is a true classic among models, as it has many cool properties not availible for other models, such as:
- interpretability
- fast learning
- precise control over underfitting/overfitting
- informational measure of importance of features
- can learn both regression and classification

Later this model is still used today as part of some ensemble methods (such as gradient boosting and random forest).

Idea of the model is to build a tree over dataset, where each node of tree learns some sort of splitting rule, usually formulated as border value for continuous variable and equility for categorial. So every node learns feature *f* and splitting value *v*.

To define the best split, a simple idea is used: the better split works, the more defined splitted halfes are. Formally speaking, it can be:

- entropy
- gini

for categorial target and

- variance
- mad_median

for cotinuous target (regression problem). All of these are implemented and can be used.