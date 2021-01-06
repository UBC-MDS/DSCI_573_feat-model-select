# DSCI 573: Feature and Model Selection

This course is about evaluating and selecting features and models. It covers the following topics: evaluation metrics, feature engineering, feature selection, the role of regularization, loss functions, and feature importances. 

2020-21 course instructor: **Varada Kolhatkar**

## Course Learning Outcomes

By the end of the course, students are expected to be able to
- build, debug, appropriately evaluate, and refine supervised machine learning models
- reason to some extent the choice of a machine learning model
- explain different feature selection methods and carry out feature selection 
- broadly describe and carry out feature engineering
- explain and carry out L1- and L2-regularization

## Deliverables
    
The following deliverables will determine your course grade:

| Assessment       | Weight  | 
| :---:            | :---:   |
| Lab Assignment 1 | 15%     |
| Lab Assignment 2 | 15%     |
| Lab Assignment 3 | 15%     |
| Lab Assignment 4 | 15%     |
| Quiz 1           | 20%     |
| Quiz 2           | 20%     |

## Class Meetings

We will be meeting three times every week: twice for lectures and once for the lab. 

### Lecture format
 
- Lectures of this course will be a combination of pre-recorded videos and class discussions and activities. The night before each lecture, the material will be made available to you in the form of pre-recorded videos and Jupyter notebooks. You'll be given time during the lecture to watch lecture videos.
- For each lecture I'll open a question thread as an issue in the course repository. Post your questions on the lecture material in this thread. 

## Lecture Schedule

Videos of this course will be available [here via Canvas](https://canvas.ubc.ca/courses/59091) and [here via OneDrive](https://ubcca-my.sharepoint.com/:f:/g/personal/varada_kolhatkar_ubc_ca/EohSYm1MzKROtq7_BTbOHTEBZ-CDuMOW51bnBMQuTSSEeg?e=Vq5PIn). 

| Lecture  | Topic  | Dataset | Resources and optional readings |
|-------|------------|-----------|-----------|
| 1     | [Evaluation metrics for classification, class imbalance](lectures/01_lecture-classification-metrics.ipynb)  | [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)| <li>[The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf)</li><li>[PR curve are better than ROC for imbalanced datasets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)</li><li>[ROC animation](https://github.com/dariyasydykova/open_projects/tree/master/ROC_animation)</li>|
| 2     | [Ridge and evaluation metrics for regression](lectures/02_lecture-ridge-reg-metrics) | [Kaggle House Prices data set](https://www.kaggle.com/c/home-data-for-ml-course/)| |
| 3     | [Feature engineering](lectures/03_lecture-feature-engineering.ipynb) |[MARSYAS](http://marsyas.info/downloads/datasets.html) | <li>[spaCy](https://github.com/explosion/spaCy)</li><li>[The Learning Behind Gmail Priority Inbox](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36955.pdf)</li><li>[Google n-gram viewer](https://books.google.com/ngrams)</li><li></li>|
| 4     | [Feature selection](lectures/04_lecture-feat-importances-selection.ipynb) | [Kaggle House Prices data set](https://www.kaggle.com/c/home-data-for-ml-course/) | |
| 5     | [Ensembles](lectures/05_lecture_ensembles.ipynb) | [Adult census data set](https://www.kaggle.com/uciml/adult-census-income) | |
| 6     | [Feature importances and loss functions](lectures/06_lecture-shap-loss-functions.ipynb) | [Adult census data set](https://www.kaggle.com/uciml/adult-census-income) | <li>[SHAP](https://github.com/slundberg/shap)</li><li>[Mike's video](https://www.youtube.com/watch?v=OqakHTDV3iI&feature=youtu.be&t=2812)</li> |
| 7     | [Regularization](lectures/07_lecture-l2-l1-regularization.ipynb)| | |
| 8     | [More linear classifiers](08_lecture-more-logistic-regression.ipynb) | | |


### Installation
We are providing you with a `conda` environment file which is available [here](env-dsci-573.yaml). You can download this file and create a conda environment for the course and activate it as follows. 

```
conda env create -f env-dsci-573.yaml
conda activate 573
```
In order to use this environment in `Jupyter`, you will have to install `nb_conda_kernels` in the environment where you have installed `Jupyter` (typically the `base` environment). You will then be able to select this new environment in `Jupyter`.

Note that this is not a complete list of the packages we'll be using in the course and there might be a few packages you will be installing using `conda install` later in the course. But this is a good enough list to get you started. 

## Reference Material

#### Books
* [A Course in Machine Learning (CIML)](http://ciml.info/) by Hal Daumé III (also relevant for DSCI 572, 573, 575, 563)
* Introduction to Machine Learning with Python: A Guide for Data Scientists by Andreas C. Mueller and Sarah Guido.
* [The Elements of Statistical Learning (ESL)](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
* [ML:APP](http://www.cs.ubc.ca/~murphyk/MLbook/index.html), 
* [LFD](http://amlbook.com/), 
* [AI:AMA](http://aima.cs.berkeley.edu/)
* [Feature Engineering and Selection: A Practical Approach for Predictive Models](https://bookdown.org/max/FES/)
* [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf)

#### Online courses

* [Mike's CPSC 330](https://github.com/UBC-CS/cpsc330)<br>
Mike is currently teaching an undergrad course on applied machine learning. Unlike DSCI 571, CPSC 330 is a semester-long course but there is a lot of overlap and sharing of notes between these courses. You might find the course  useful.  
* [Mike's CPSC 340](https://ubc-cs.github.io/cpsc340/)
* [Machine Learning](https://www.coursera.org/learn/machine-learning) (Andrew Ng's famous Coursera course)
* [Foundations of Machine Learning](https://bloomberg.github.io/foml/#home) online course from Bloomberg.
* [Machine Learning Exercises In Python, Part 1](http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/) (translation of Andrew Ng's course to Python, also relevant for DSCI 561, 572, 563)

#### Misc

* Python code associated with _An Introduction to Statistical Learning_
    - [Jupyter Notebooks from @mscaudill](https://github.com/mscaudill/IntroStatLearn)
    - [Jupyter Notebooks from @JWarmenhoven](https://github.com/mscaudill/IntroStatLearn)
* [R code associated with _An Introduction to Statistical Learning_](http://www-bcf.usc.edu/~gareth/ISL/code.html)
* [The Art of Feature Engineering](http://artoffeatureengineering.com/)
* [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
  
## Policies

Please see the general [MDS policies](https://ubc-mds.github.io/policies/).
