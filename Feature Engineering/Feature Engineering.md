
Wrapper Methods
- external methods to choose a features rather than as port of the model process
	- Can result in overfitting

Embedded Methods
- Models where the feature selection procedure occurs natural course of the model fitting process.
	- Ex: Decision Trees

Unsupervised selection methods can hbave positive effect on model performance


Example: Predict Patient risk for ischemic stroke
- Degree of arterial stenois (blockage) is an indictor
- Over 70%
- Alone not a good indicator

Steps
1. Splitting the Data
	1. Strattegified Data
		1. Equal outcomes in test and training dataset
		2. Keeps proportions equal
2. Preprocessing
	1. Missing values
		1. Replaced with median value
	2. Mean Center Some data
	3. Skewness?
		1. Log transform or Box-Cox or Yeo-Johnson transform
	4. Remove predictors with high R^2 with others
		1. >.9
3. Explorations
	1. Resampling
		1. 10 fold cross validation
			1. 50 variations of training set are conducted
			2. Evaluated on one of these resample
			3. Some protection against overfitting
	2. Compare 2 models:
		1. For 1: for each resample do
			Use the resample's $90 \%$ to fit models $M_1$ and $M_2$
			Predict the remaining $10 \%$ for both models
			Compute the area under the ROC curve for $M_1$ and $M_2$
			Determine the difference in the two AUC values end for
			Use a one sided t-test on the differences to test that $M_2$ is better than $M_1$.
	3. Look at pairwise interactions between terms
		1. Plot ROC improvement due to interaction vs log of p value
4. Predictive Modelling
	1. However, it is well known that logistic regression is a high-bias, low-variance model which has a tendency to yield lower predictivity than other low-bias, high-variance models.
		1. The predictive performance of logistic regression is also degraded by the inclusion of correlated, non-informative predictors. In order to find the most predictive logistic regression model, the most relevant predictors should be identified to find the best subset for predicting stroke risk.
	2. Yeo-Johnson transformation as well as centering and scaling. for predictgor then run a Recurive Feature elimination
	3. orrelation between the predictors can cause instability in the logistic regression coefficients. While there are more sophisticated approaches, an additional variable filter will be used on the data to remove the minimum set of predictors such that no pairwise correlations between predictors are greater than 0.75.