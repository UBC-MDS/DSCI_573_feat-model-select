# DSCI 573: Feature and Model Selection

How to evaluate and select features and models. Cross-validation, ROC curves, feature engineering, the role of regularization. Automated hyperparameter optimization.

## Schedule

Relevant textbooks: [AI:AMA](http://aima.cs.berkeley.edu), [ESL](https://web.stanford.edu/~hastie/ElemStatLearn), [ML:APP](http://www.cs.ubc.ca/~murphyk/MLbook/index.html), [LFD](http://amlbook.com/) 


| #  |  Topic   | Related (Optional) Reading |
|-----|--------|-----------|
| 1 | [Fundamentals](lectures/S1.pdf?raw=1)  | [Overfitting](https://en.wikipedia.org/wiki/Overfitting) [Data Dredging](https://en.wikipedia.org/wiki/Data_dredging) [Cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) AI: AMA 18.4-5, ESL 7.1-7.4, 7.10, ML:APP 1.4, 6.5 [Fluid Simulation](https://www.inf.ethz.ch/personal/ladickyl/fluid_sigasia15.pdf)|
| 2 | [Nonlinear Regression](lectures/S2.pdf?raw=1)  | ESL 5.1 |
| 3 | [Feature Selection](lectures/S3.pdf?raw=1) | [kaggle writeup](http://blog.kaggle.com/2016/04/08/homesite-quote-conversion-winners-write-up-1st-place-kazanova-faron-clobber/)  [Genome-Wide Assocation Studies](https://en.wikipedia.org/wiki/Genome-wide_association_study) [AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion) [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion) ESL 3.3, 7.5-7 |
| 4 | [Feature Engineering](lectures/S4.pdf?raw=1) | [Wikipedia](https://en.wikipedia.org/wiki/Feature_engineering) |
| [Quiz Practice Questions](quiz_practice_questions.pdf) | L1-4 | |
| 5 | [Regularization](lectures/S5.pdf?raw=1) | ESL 3.4., ML:APP 7.5, AI:AMA 18.4  |
| 6 | [Linear Classifiers](lectures/S6.pdf?raw=1) | [RBF Video](https://www.cs.ubc.ca/~schmidtm/Courses/340-F16/rbf.mp4) [RBF and Regularization Video](https://www.cs.ubc.ca/~schmidtm/Courses/340-F16/rbf2.mp4) ESL 6.7, ML:APP 13.3-4   |
| 7 | [Boosting](lectures/S7.pdf?raw=1) | ESL 4.4, ML:APP 8.3.7, 9.5 [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost) [XGBoost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html) |
| 8 | [MLE/MAP](lectures/S8.pdf?raw=1) | [Wikipedia](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) ML:APP 9.3-4  |
| [Quiz 2 Practice Questions](quiz2_practice_questions.pdf) | L5-8 | |


## Resources

- [Feature Engineering and Selection: A Practical Approach for Predictive Models](https://bookdown.org/max/FES/)
- [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf) by Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani, specifically:
    - Chapter 2: Statistical Learning  
    - Chapter 5: Resampling Methods
    - Chapter 6: Linear Model Selection and Regularization
    - Chapter 7: Moving Beyond Linearity
- Python code associated with _An Introduction to Statistical Learning_
    - [Jupyter Notebooks from @mscaudill](https://github.com/mscaudill/IntroStatLearn)
    - [Jupyter Notebooks from @JWarmenhoven](https://github.com/mscaudill/IntroStatLearn)
- [R code associated with _An Introduction to Statistical Learning_](http://www-bcf.usc.edu/~gareth/ISL/code.html)
