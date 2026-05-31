## Overview
Overview: Complex models make it difficult to balance interpretability and performance. Reducing the number of features may help this issue. The motivation for removing predictors from a model should be to reduce model complexity or mitigate a specific problem.
- SVM and Neural Networks are sensitive to irrelevant predictors
- Linear regression and Logisitc regression are vulnerable to correlated predictors. Removing correlated predictors can reduce multicollinearity and enable models of these type to be fit.

So Goal becomes: Reduce the number of predictors as far as possible without compromising performance.

3 main methods of Feature Selection Exist
- Intrinsic
	- Types:
		- Tree and Rule Based Models
			- If a predictor is not used in any split, then it is not predictive
			- RF, however forces splits on irrelevant features resulting in overselection of predictors
			- Trees: use Greedy Search which may have a narrow set of features that have sub-optimal performance.
		- Multivariate Adaptive Regression Spline (MARS) models
			- Create new features that involve on or two predictors at a time. If a predictor is not involved in at least one MARS feature it is , excluded from the model
		- Regularization models
			- Regularization approach penalizes or shrinks predictor coefficients to improve model fit. ex: Lasso
	- Relatively fast because it is built into model selection process.
	- Direct connection between feature selection and objective function
	- Downside: feature selection is model dependent. May be better to use non-intrinsic selection. 
- Filter
	- Conduct an initial supervised analysis to determine which features are important. 
	- Each feature is evaluated separately
		- ex: log odds ratio and statistical significance
	- unlikely to capture independent trends in the data
	- Good at capturing long trends (individual predictor-outcome relationship). Prone to over selection.
		- Statistical significance may not lead to predictive performance
- Wrapper
	- Iterative search procedures that repeatedly supply predictor subsets to the model and then use the resulting model performance estimate to guide the slection of the next subset to evaluate.
	- Can be greedy or non-greedy approaches.
		- Greedy is best at current time. Non-greedy not necessarily best now but possibly better affects down the road.
	- Benefit: Ability to search through a wider variety of predictor subsets than simple filters or models with built-on feature selection
		- Best potential to find global minimum
	- Drawback: Computation time, potential for overfitting
	- Example: Recursive Feature Selection
		- Greedy which start will all features then reduces them down slowly one at a time.
		- Will fail if there are important interactions present
	- Example: Genetic algos and simulated annealing(uses randomness)

Overall Good Strategy
- Start with intrinsic to see what they yield
	- Make sure to confirm that a non-linear wrapper gos with a non-linear method

**Effect of Random Irrelevant Features**

Predictability depends on
- type of model
- nature of predictors
- ratio of size of training set to number of predictors

![[Effect of Irrelevant Features.png]]

- Lasso: Doesn't Care about extra features from GLMNET
- MARS was good in both cases
- SVM and neural networks were both good but more irrelvant features detroys performance
- Trees don't care about extra features for perofrmance
	- But will include them in feature selection, vastly overselects features
	- Main paramter $m_{try}$ number of randomly selected features that should be evaluated at each split

Solution to overfitting?
- Evaluate tuning parameters on a validation set (not the testing set)
- Feature selection needs to be part of resampling process
- Common Mistake: Conduct resampling inside the feature selection process
	- 1:Ex: Rank the predictors using the training set
	- 2:for a feature subset of size 5 to 1 do
		- 3:for each resample do
			- 4:Fit model with the feature subset on the analysis set
			- 5:Predict the assessment set
		- 6:end for
	- 7:Determine the best feature subset using resampled performance
	- 8:Fit the best feature subset using the entire training set
9:end for

Issues with this?
- Feature selection is external to resampling. Resampling cannot measure the impact of the selection process.
- Same data is being used to measure performance and guide the direction of the selection routine. Some sort of _out-of-sample_ data are required to accurately determine how well the model is doing.


Better way of combining feature selection and resampling: Make feature selection a component of the modelling process.

```
Split data into analysis and assessment sets;
for each resample do
    Rank the predictors using the analysis set
    for a feature subset of size 5 to 1 do
        Fit model with the feature subset on the analysis set
        Predict the assessment set
    end for
    Average the resampled performance for each model and feature subset size
    Fit the best feature subset using the entire training set
    Choose the model and feature subset with the best performance
    Fit a model with the best feature subset using the entire training set
end for
```
- Same data is not used to determine subset size and predictors within the subset
- External loop
	- HHow far down the removal path the selection process will go
	- This then determines the subset size that should be used in the final model
	- Subset size is a tuning parameter
- Implications
	- More real estimate of predictive performance

# Contents
1) Greedy Search Methods
	1) Simple Filters
	2) Recursive Feature Elimination
	3) Stepwise Selection
2) Global Search Methods
3) Fano's Inequality - High Mutual Information

## 1) Greedy Search Methods

**Simple Filters**
Easiest way to perform feature selection: Score the predictors relationship with the outcome variable. How to score them?
- Variable: Categorical
	- Outcome: Categorical 
		- 3 or more $\chi^2$ tests
		- 2: odds ratio
	- Outcome: Numeric
		- 3 or more: traditional ANOVA $F$-statistic
		- 2: $t$ test, ROC, precission-recall curves
- Variable: Numeric
	- Outcome: Categorical
		- Same tests,
		- Large correlation or multi collinearity: correlation adjusted t score 
	- Outcome: Numeric
		- Correlation statistic
		- If nonlinear, maximal information coefficient
		- Alternatively: GAM can fit nonsmooth terms to a set of predictors smultaneously and measure their importance using a p value that tests a null hypothesis that there is no trend of each predictor
- What to do in mixed data?
	- Use  a permutation metric to convert any metric to a p value
	- p values are not meaningful indicators of statistical significance
	- How to handle if a feature has a good p value with the current data but not good with new data?
		- Increase the p value
- We need to perform two layers of cross validation
	- 1) used to filter feature
	- 2) use to select tuning parameters
	- ![[Training data split for filtering and tuning params.png]]
	- ex:
		- For filtering methods use an ROC curve on a PLSDA model
		- 20 iterations of bootstrap resampling for internal sampling and 5 repeats of 10 fold cross validation for external resampling
		- Final estimates of performance based on external resampling hhldoujt out sets
		- Yeo-Johnson transformation
Post Hoc Anaylsis: advisable when conducting feature selection is to determine if the particular subset that is found is any better than a random subset of the same size.
- Take 100 random sets of 5 predictors and fit them on the same routine!
The drawback is that there may be some redundancy in the selected features and the subjectivity around the filtering threshold might leave the modeler wanting to understand how many features _could_ be removed before performance is impacted.

**Recursive Feature Selection**

Start with model with all predictors and then remove the least important predictor and compute importance scores again.
- Subset Size is a tuning parameter
	- subset size that optimizes the performance criteria is used to select the predictors based on the importance rankings
	- The selection process is resampled in the same way as fundamental tuning parameters from a model, such as the number of nearest neighbors or the amount of weight decay in a neural network. The resampling process _includes_ the feature selection routine and the _external_ resamples are used to estimate the appropriate subset size.
- Can't be used with certain models where number of predictors exceed sample size
	- Logistic, Multiple, LDA
	- RF actually benefits from this
- Backward Selection use used with Random Forest often
	- RF tends not to exclude variables from the prediction equation
		- averaging models that are effectively the same does not drive down the variation in the model predictions.
		- The act of restricting the number of predictors that could possibly be used in a split increases the likelihood that an irrelevant predictor will be chosen.
		- RF can use some post hoc pruning of variables that are not essential for model to perform well
	- Model has a well known internal method for measuring feature importance
		- RF has an issue where multicollinearity exists RF randomly picks a feature
		- Redundant features lowers importance score of a feature
	- Advice: Expierment with filtering out highly correlated feature prior to beginning the routine


**Stepwise Selection**

AVOID THIS PROCEDURE!!!!
- use glmnet or bayesian methods!
Entry and exit from the model is usually based on $p$ values
- <.15 or >.15 to leave

Steps
- Create $p$ lr model with use exactly one feature each
- The importance of the features are then ranked by their individual ability to explain variation in the outcome.
- If a value has p values less than .15 then process continues and have $p-1$ models
	- else we stop
Why is this approach bad?
- violates statistical and hypothesis testing
- 1) inflation of positive findings
	- The corresponding p-values are unadjusted, leading to an over-selection of features
	- Solution
		- Use AIC: nd penalizes the likelihood by the number of parameters included in the model.
		- Therefore models that optimize the likelihood and have fewer parameters are preferred.
- 2) overfitting
	- The resulting model statistics, including the parameter estimates and associated uncertainty, are highly optimistic since they do not take the selection process into account. In fact, the final standard errors and p-values from the final model are incorrect.
- You need external resampling to avoid 3!!!
- 
## 2) Global Search Methods
## 3) Fano's Inequality

A common method for feature selection is to pick input features $X_d$ which have high mutual information with response variable $Y$
- Fano's inequality: bounds the probability of misclassification in terms of mutual information between the features $X$ and class label $Y$

Theorem 6.3.2. (Fano's inequality) Consider an estimator $\hat{Y}=f(X)$ such that $Y \rightarrow X \rightarrow \hat{Y}$ forms a Markov chain. Let $E$ be the event $\hat{Y} \neq Y$, indicating that an error occured, and let $P_e=P(Y \neq \hat{Y})$ be the probability of error. Then we have

$$
\begin{equation*}
\mathbb{H}(Y \mid X) \leq \mathbb{H}(Y \mid \hat{Y}) \leq \mathbb{H}(E)+P_e \log |\mathcal{Y}| 
\end{equation*}
$$

Since $\mathbb{H}(E) \leq 1$, as we saw in Figure 6.1, we can weaken this result to get
$$
\begin{equation*}
1+P_e \log |\mathcal{Y}| \geq \mathbb{H}(Y \mid X) 
\end{equation*}
$$
and hence
$$
\begin{equation*}
P_e \geq \frac{\mathbb{H}(Y \mid X)-1}{\log |\mathcal{Y}|} 
\end{equation*}
$$

Thus minimizing $\mathbb{H}(Y \mid X)$ (which can be done by maximizing $\mathbb{I}(X ; Y)$ ) will also minimize the lower bound on $P_e$.
- In statistics:
	- each hypothesis corresponds to a probability distribution
	- if distributions overlap too much,  no estimator can distinguish them
- Fano’s inequality says that if many hypotheses produce statistically similar data, then no algorithm can reliably determine which hypothesis generated the observations.
- If mutual information is large: then the lower bound on error decreases. So high mutual information makes low classification error _possible_.
- Mutual Information is $I(X ; Y)=H(Y)-H(Y \mid X)$
	- $X$ is features
	- $Y$ is class labels
	- $H(Y)$: Entropy: uncertaintity about class before seeing features
	- $H(Y |X)$: uncertainity remaining after seeing features
	- $I(X;Y)$ measures how informative features are about the label

```

from sklearn.feature_selection import mutual_info_classif  
mi = mutual_info_classif(X, y)  
print(mi)

# Selects the 10 best ones that 
from sklearn.feature_selection import SelectKBest  
selector = SelectKBest(mutual_info_classif, k=10)  
X_selected = selector.fit_transform(X, y)
```
```
```

