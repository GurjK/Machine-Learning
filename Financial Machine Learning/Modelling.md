## Ensemble Methods

Error in models comes from:
- Bias: unrealistic expectations. Model can't recognize the features relationship
	- Underfitting
- Variance: Sensitivity to small changes in the training set
	- Overfit
	- Mistaking signal with noise
- Noise: Error caused by the variance of the observed values like unpredictable changes

Mean Squared Error of Estimation Error: 
- $\mathrm{E}\left[\left(y_i-\hat{f}\left[x_i\right]\right)^2\right]=(\underbrace{\mathrm{E}\left[\hat{f}\left[x_i\right]-f\left[x_i\right]\right]}_{\text {bias }})^2+\underbrace{\mathrm{V}\left[\hat{f}\left[x_i\right]\right]}_{\text {variance }}+\underbrace{\sigma_{\varepsilon}^2}_{\text {noise }}$

Bootstrap Aggregation (Bagging):
- generate $N$ training datasets with random sampling with replacement
- fit $N$ estimators on on each training set
- Fit independently of each other
- Ensemble forcast is average of all the models!

Advantages of Bagging:
- Refuces forecasts'ss variance to reduce overfitting
- I $p > \frac{1}{k}$, for large $N$ then ensemble beats individual classifier
- If classifiers are independent and better than average, the bagging reduces the variance and majority voting amplifies signal
- Bagging will reduce the variance but not as much bias because it will not improve perfromance of poor classifiers

Redundant Data
- 5 minute bar at 12:00 and 12:01 has 80% of the same data so not independent
	- use sequential bootstrapping to solve this!
- This makes bagging ineffective!

#### Random Forests

Decision trees typically overfit, increasing variance of the forecasts.
- Random Forest was designed to fix this

Random forests
- Second source of ranodmoness: when optimizing on a node split, it was sample a random subsample of features without replacement of the attributes with purpose of decorrelating estiamtors
- Determines feature importance!
- Not necessarily going to have lower bias than individual trees
- Need IID to make these affective! 
	- Without we get, high correlation between trees so overfitting!
- RF fixes the size of the bootstrapped samples to match the size of the training dataset!

In SKLearn
1. Set max_features to a lower values
	- Reduces the subset of features. Reduces the correaltion accross trees
2. Ret regularization parameter: min_weight_fraction_leaf = 5% or higher such that out of bag accuracy converges to out of sample k fold acccuracy
	- reduces the overlap effect so model is no longer overfitting to redundant patterns
3. Use BaggingClassifier on DecisionTreeClassifier where max_ samples is set to the average uniqueness (avgu) between samples.
(a) clf=DecisionTreeClassifier(criterion='entropy',max_ features='auto',class_weight='balanced')
(b) bc=BaggingClassifier(base_estimator=clf,n_estimators= 1000, max_samples=avgU, max_features=1.)
	- Sample on the indepdent portions of the data so u get less overlap inside each tree
4. Use BaggingClassifier on RandomForestClassifier where max_ samples is set to the average uniqueness (avgu) between samples.
	1. (a) clf=RandomForestClassifier(n_estimators=1,criterion= 'entropy',bootstrap=False,class_weight='balanced_ subsample')
	2. (b) bc=BaggingClassifier(base_estimator=clf,n_estimators= 1000 , max_samples=avgU, max_features=1.)
	- Use a random forest with 1 tree then a bagging classifier. RF handles feature randomness and Baggng handles sample randomness. Standard RF using naive bootstrap which is abd for overlapping data. 
5. Modify the RF class to replace standard bootstrapping with sequential bootstrapping.
	1. Use newer information to penalize overlap and prefer unique observations

Suggestion:
- Do RF on a PCA of features as tis will reduce number of levels needed. 

1. What Out-of-Bag (OOB) accuracy is

In bagging (and Random Forests):
- Each tree is trained on a bootstrap sample (sampling with replacement)
- On average:
- $\sim 63 \%$ of observations are used in training
- $\sim 37 \%$ are left out → these are out-of-bag samples
✓ For each observation:
- You average predictions from trees where it was NOT used in training

That gives:
OOB accuracy $\approx$ internal "test set" without explicit CV
2. What Stratified K-Fold (with shuffling) is

In stratified k -fold CV :
- Data is split into $k$ folds
- Each fold:
- preserves class proportions
- With shuffling:
- data is randomly mixed before splitting

You train on $k-1$ folds and test on the remaining one, rotating.

Bagging for Scalability
- Base estimator make it something that does not scale well with size SVM
	- impose tight early exiting conditions
- Early stopping will increase variance of estimator outputs
- however, that increase can be more than offset by the variance reduction asso ciated with the bagging algorithm. You can control that reduction by adding more independent base estimators. Used in this way, bagging will allow you to achieve fast and robust estimates on extremely large datasets.
#### Boosting

![[ADABoost.png]]

Boosting
- Individual classifers are fit sequentially
- Poor-performing classifiers are dismissed
- Observations are weighted differently in each iteration
- Ensemble forecast is a weighted average of the individual learners

Boosting Benefits
- Reduces variance and bias!
	- Greater risk of overfitting added
- But easy to overfit financial data so bagging may be better!
## Cross Validation in Finance

Goal: Determine Generalization error of an ML Algo

Setup K fold CV:
- Split data into $k$ data sets
- for $i=1,...,k$
	- train on everything but i and test on i
- Outcome is a $k \times 1$ array of cv performance metrics

In financial situations used in
- hyper parameter tuning (model development)
- backtesting

Why does K fold CV fail in finance
- Observations can't be assumed to be iid
- testing set is used multiple times in the process of developing a model, leading to multiple testing and selection bias

- Serial Correlation causes: $X_t \approx X_{t+1}$
- Labels are derived from overlapping datapoints so same as above for $y$
	- Placing $t$ and $t+1$ in different sets means information is leaked
	- If $X_t$ is a predictive feature then leakage will enhance performance of strategy!

Two Ways to reduce Leakage
- Drop from training set any observation $i$ where $y_i$ is a function of information used to determine $Y_j$ and $j$ belongs to test set
	- $Y_i$ and $Y_j$ should not span overlapping periods
- Avoid overfitting the classifier
	- Early Stopping of base estimators.
	- Basggins of classfiers, while controlling for oversampling on redundant examples so individual classifiers are diverse as possible
		- set max_samples = average uniqueness
		- Apply sequential bootstrap

Purged K-Fold CV

Goal: Remove from the training set all observations whose labels overlapping in time with those labels included in test set
- Also remove observations that immediately follow an observation in test set!
- Training starts within test
- Train ends within test
- Train envelops test!!

So in the end
- Train -> purge -> test -> purge -> train
	- purges are the overlaps!!!
	- If a testing label is determined by price action from Monday to Friday, and a training label is based on price action starting on Wednesday, they share common price returns from Wednesday through Friday
- Maybe need an embargo too after each test Set! $h$
	- Let $h = .01T$ to suffice for all leakage
	- Train -> purge -> test -> embargo + purge -> train
- In general, we need to purge and embargo overlapping training observations whenever we produce a train/test split, whether it is for hyper-parameter fitting, backtesting, or performance evaluation.

Ex: predicting 7 days: 

**K fold CV: Is used to validate a model that will go into a backtest. Model evaluation/tuning tool and not a trading tool**
- Use for tuining parameters, select features, and compare models!!
## Feature Importance

Repeating a backtest over and over again is for example overfitting!!!

Setup: Mattrix with $X$: features and $y$: labels for a particular instrument
- Can fit through purged k-fold cross validation
- Performs well so determine which features caused this?

Once have good features determine
- What triggers importance over time?
- Features always important or only some of the time?
- Can regime switches be predicted?
- Are they relevant to other asset classes?
- What is subset of features with highest rank correlation across the entire investment space?

**Feature importance is a research too. Backtesting is not**

##### Feature importance with subsitution effects
- Mean Decrease Impurity: Fast explantory importance method specific to tree based classifiers.
	- Fast and in sample
	- MDI: **total impurity reduction contributed by that feature across the tree(s)**
	- impurity: a measure of how mixed the labels are in a subset
		- Nodes with all the same class: 0 impurity, mixed means higher
	- Decrease impurity at each tree split. 
		- Measure how much of all impurity decrease can be assigned to each feature
		- When tree splits on a feature -> impurity decrease = Parent impurity -  weighted child impurity
	- Notes:
		- Each feature is given a chance to reduce impurity and makes sure features with 0 importance are not average. replace with np.nan.
		- Each feature will have some importance since in sample
		- Can't extend to non tree based
		- The method does not address substitution effects in the presence of correlated features. MDI dilutes the importance of substitute features, because of their interchangeability: The importance of two identical features will be halved, as they are randomly chosen with equal probability.
		- Biased towards some predictor variables
- Mean Decrease Accuracy: slow predictive importance (OOS)
	- OOS: according to accraucy or some loss function
	- permutates each column of features!
	- Importance of a feature is a function of the loss of performance cause by its column's permutation
	- Notes:
		- Any classifier
		- Can be applied to anything, not just accuracy
		- Procedure is suseptible to subsittuion effects in correlated featues
		- MDA can make features look outright irrelevant , even if critical if they are two correlated features
		- CV must be purged and embargoed 

##### Feature Importance without Substitution

Substitution can lead to important features being discarded. Consider following to aid previous
- Single Feature Importance: Cross Sectional predictive importance method
	- OOS perfromance score for each feature in isolation
	- Notes:
		- Any classifier
		- Not limited to accuracy
		- No substitution takes place, only one feature at a time.
		- It can conclude that all features are important because perforamnce is evaluated via OOS CV.
	- Limitation
		- Classifier with two features can performance better than bagging of two single feature classifiers
			- Joint effects and hiearchical importance is lost!
- Orthogonal Features: Solution to underestimating importance is to orthoganlize features before applying MDI and MDA. For Example : PCA!!
	- Make sure when perfroming PCA to 
		- Center the Data: Ensures first component is in direction of observations
		- Rescale: PCA focuses on explaining correlations rather than variations!
			- No recaling means highest variances dominate!
	- Benefits:
		- can be used to reduce dimensionality of the features of the matrix $X$
		- conducted on features designed to explain the structure of the data
		- WHEN PCA and MDI/MFA/SFI both choose the same features it gives proof that we are not overfitting!
		- PLOT MDI on Y and Eigenvalyes on X axis to confirm this!!
	- Kendall's tau:
		- Compute kendalls tau between feature importance and their eigenvalues. Want this to be closer to 1!
		- Possibly use a wieghted kendall's tau

##### Parallelized vs Stacked Feature Importance
- Parallel
	- higher values across a large universe of investments means an underlying phenomenon may be in contrul
	- Derive the feature importance in parallel
		- Feature j on instrument i to criterion k
	- t is highly susceptible to **substitution effects** (multi-collinearity). Because importance is calculated per instrument, redundant features might swap ranks between different securities, which increases the variance of the final aggregated estimates
- Stacked
	- Instead of treating securities separately, this approach combines them into one large global dataset
	- Stack individual datasets into one dataset
		- First transform features  by standardizing on a rolling trailing window to ensure distributional homegenity
	- Benefits:
		- Model trainined to larger dataset
		- Importance is derived directly
		- conclusions are generalized
	- classifier can be fit on a set of instruments, including for the purpose of model predic tion. That reduces the likelihood of overfitting an estimator to a particular instrument or small dataset.

Toy Example:
- Data
	- Redundant (linear combinations of infromative features causing subsitution effects), Infromative for explaining, Noise
- Results
	- MDI: Substitution effects cause some informative or redundant features to rank better than others, which was expected.
		- R and I are most important: N are all not important
	- MDA: Similar to MDI
	- SFI: Some features perform worse than noise!
## Hyper Parameter Tuning with Cross Validation

Grid Search Cross Validation: Exhaustive search for combination of parameters that maximize the CV performance
- Scikit: GridSearchCV
	- Need to implement a purgested Grid Search CV!!!
	- for metalabelling using scoring  = F1
	- other application use: accuracy or negative log liklihood
		- for hyperparemter use negative log lilklihood!
		- Accuracy does not account for the confidence of a predition!
		- Log Lillihood takes into account the probabilities which matters for us!!
		- CV Loss > Accuracy:
			- Takes into account probabilties
			- measures performance in terms of variables related to PNL
				- Label: Side
				- Probabilities: Position size
				- Sample weight: obersvation's returns
- Log Uniform distribution: Typically only accept positive numbers for hyperparams
	- don't use uniform to generate this but rather log uniform!
