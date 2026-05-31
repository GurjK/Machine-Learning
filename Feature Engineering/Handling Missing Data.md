
1. Understanding Missing Data
2. Models that are Resistant to Missing Values
3. Deletion of Data
4. Encoding Missingness
5. Imputation Methods
6. Special Cases

### 1) Understanding Missing Data

Methods to visual missing data
- Heatmap
- co-occurrence plot
	- displays frequency of missing predictor combinations
- PCA
	- Predictor matrix converted to binary matrix where a zero means non-missing and one means missing
	- intiial dimensions capturing variations would be caused by presence of missing values
	- samples with no missing values will probably be close to same location, close to origin
	- plot the scores of first two dimensions we can identify nature and degree of missing ness in data
	- Samples must be columns and rows are predictors
- Numerical statistics

## 2) Models that are Resistant to Missing Values

Models that can't handle missing data
- neural networks
- glmnet
- svm

Models that can handle missing data
- CART: uses surrogate splits
- C5.0: Based on distribution, fractional counts are used in following splits
- Naive Bayes: class specific distibutions of each predictor separately!

## 3) Deletion of Data

Options
- Remove entire predictor and/or sample that can missing values
	- remove predictors  that contain at least one missing value, removing all samples with any missing data, Similarly, missing values could be eliminated by removing all samples with any missing values.
- Consider: Intrinsic value of samples as compared to predictors
	- Low Samples, don't want to delete data
	- Samples in general are more valuable so deleting them is NOT good practice
	- Initially, remove preditors that have a large amount of missing data
	- Ex: highly correlated predictors, probably okay to drop one of them if it has missing data
- By throwing away data, we might bias the model as it relates to the predictor to the outcome.

## 4) Encoding Missingness

- Encode when a variable is discrete as "unknown" or "missing" or "no alley"
- A guiding principle that can be used to determine if encoding missingness is a good idea is to think about how the results would be interpreted if that piece of information becomes important to the model.

## 5) Imputation Methods

One method: Impute or Estimate Missing data. 

Statistical Imputation
- The primary approach in this scenario is to use multiple imputations; several variations of the data set are created with different estimates of the missing values. The variations of the data sets are then used as inputs to models and the test statistic replicates are computed for each imputed data set. From these replicate statistics, appropriate hypothesis tests can be constructed and used for decision making.

Inferential vs Predictive models
- Predictive models can't produce inferential results like in tree models.
- To ensure that the variation imparted by the imputation is captured in the training process, we recommend that imputation be performed within the resampling process.
- Important that imputed values be as close as possible to their true (unoberseved) values
- Inferential methods want to understand relationships between predictors and response
	- Conversely, the focus of predictive models is to understand the relationships between the predictors and the response that are generalizable to yet-to-be-seen samples.
	- Multiple imputation methods do not keep the imputation generator after the missing data have been estimated which provides a challenge for applying these techniques to new samples

Goal is to produce most accurate prediction of the missing data point
- Should be fast and have compact prediction equation
- should be tolerant of other missing data
- able to use predictors of various types of input
- should be stable and not overly influenced by outliers

General Guidance:
- 20% is max of missing column
- Imputation is first step in any preprocessing sequence

Models
- K-Nearest Neighbors: Average of K nearest neighbors measured by distance
	- Distance can be Euclidian or Gower Distance
- Trees: Random Forests
	- Drawback: Need 500 to 2000 trees to achieve stable, reliable model
	- Alternative, Bagged Tree:
		- Difference: All predictors are evaluated at each split in each tree
		- Bagged Tree is usually in range of 25-50 trees
- Linear Methods: When a complete predictor shows a strong linear relationship with a predictor that requires imputation, a straightforward linear model may be the best approach.
	- Numeric: Linear
	- Qualitative: Logistic

## 6) Special Cases

Data can be not completed but also not missing
- Ex: when duration of time until event $T$ but $T$ has not occured yet
- For left censored when you don't know lower bound of $X$ use a random uniform generator between 0 and $X$

If data has a strong time component
- simple moving average smoothers can be used to impute the data