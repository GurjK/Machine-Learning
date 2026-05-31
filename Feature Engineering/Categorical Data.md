
# Contents
1. One Hot Encoding: Unsupervised
	1. One 
2. Encoding Predictors with Many Categories: Unsupervised
3. supervided encoding methods for categorical data?
	1. Target Encoding or Likelihood Encoding
	2. Target Encoding with Shrinkage / Empirical Bayes
	3. Learned Embeddings (Deep Learning)
4. Encodings for Ordered Data
5. Creating Features from Text Data
6. Other:
	1. Factors Versus Dummy Variables in Tree-Based Models




## 1) One Hot Encoding

**Dummy one Hot Encoding**
- This involves creates $C-1$ variables for $C$ categories rather than full one hot encoding
- Create dummy variable =1 or 0 depending on if condition is meant
	- 1 and 0 changes interpretation of coefficients

$y_i=\beta_0+\beta_1 x_{i 1}+\beta_2 x_{i 2}+\epsilon_i= \begin{cases}\beta_0+\beta_1+\epsilon_i & \text { if } i \text { th person is from the South } \\ \beta_0+\beta_2+\epsilon_i & \text { if } i \text { th person is from the West } \\ \beta_0+\epsilon_i & \text { if } i \text { th person is from the East. }\end{cases}$
- In this method then one category is omitted
- This is good to fix multicollinearity
	- Suppose there was no North, then South + West + East would =1
	- This would introduce multicollinearity so the model would crash!!!
	- But now we say we have South, West , and East and use two dummy variables
		- East =0,0, West = 0,1, and South = 1,0
		- $\beta_0$ : mean for East (baseline)
		- $\beta_1$ : difference between South and East
		- $\beta_2$ : difference between West and East
	- If model includes intercept and all variables included then the sum of variables equals the intercept and this would mean we have a singular matrix
		- Overdetermined or less than full rank or singular
- Use Reduced One Hot Encoding when required linear regression and matrix inversion is required
- Interpretation of dummy variables
	- coefficients are difference of meme and reference variable that were included
	- intecept: left out value
	- coefficient: amount greater than reference value
	- coefficient+intercept: true value of coefficient variable

**Full One Hot Encoding**
- **One Hot Vector**: Reality this means creating a vector where everything as 0 except the row that corresponds to that category
- Using One Hot encoding allows us to write the distribution as  $\operatorname{Cat}(\boldsymbol{y} \mid \boldsymbol{\theta}) \triangleq \prod_{c=1}^C \theta_c^{y_c}$
	- Categorical Distribution: Generalizes Bernouli to >2. 
	- Can be thought of as multinomial distributions
	- Direct probability of each class
	- Applies to unordered categories
		- no hierarchy, i.e. good or bad connotation
- When to use Full One Hot Encoding? 
	- The key is: **the dummy variable trap only applies in specific models (like standard linear regression with an intercept).** Outside of that setting, full one-hot encoding is often the _natural_ or even _required_ representation.
		- 1) When you don't include an intercept
		- 2) In probablistic Models
		- 3) Classification Models
		- 4) Tree Based Models
	- Full one-hot encoding:
		- treats all categories **symmetrically**
		- doesn’t force a “baseline” category
- When not to use One Hot Encoding?
	- The number of categories $C$ is large
		- ex: zip codes in the United States
			- Resampling would excude some of the rarer categories in analysis set
			- dummy variable would be all 0s
			- new predictions that contain sample, will not provide good prediction
			- called zero variance predictor: where predictor contains a single value
		- Would cause an overdetermined or restrictive model

## 2) Encoding Predictors with Many Categories

Solutions to Issues
1) Create Full set of dummy variables and simply remove the zero-variance predictors
	1) There may be a different number of model parameters across resamples actually propagating noise into the resampling estimates of performance
		1) These end up creating constants
		2) Freq Ratio: Most common value count / second most common value count, huge ratios means extremely rare
		3) Some fold in k fold validation will not even contain the rare category!
	2) Determine which variables might be near zero variance before resampling
		1) Have few unique values
		2) occur infrequently in data
2) To avoid filtering, redefine predictor's categories prior to creating dummy variables
	1) Create "other" category to pull together rare occurrences
	2) Use train set frequency to determine the cutoff
3) Hashing function
	1) Main Benefit:
		1) Compresses categories into a fixed feature space using hashing
		2) Good for : user ids, urls, search queries, product ids 
		3) bas for:categories that are small and stable and interpretability matters alot
	2) Map one set of values to another
	3) Usually map categories to another as a power of 2
	4) unidirectional so there is no way of know the original values
	5) Some of original categories will be mapped to same hash value. The number of "collisions" is determined by number of features produced
		1) Collisions lose information 
	6) Called "feature hashing"
	7) Method
		1) Integer Division, is you want 16 columns then you do columns = (integer mod 16) + 1
		2) You may get columns that are all 0s since this is dependent on mod
		3) If a column has 2 values of 1 then , In statistical terms, these two categories are said to be _aliased_ or _confounded_, meaning that a parameter estimated from this hash cannot isolate the effect of either city.
	8) Could also make them signed, so produce values of +1,0,-1
		1) Turn collisions to be +1 and -1 instead of 2 +1s
	9) Why do we want to reduce Alliasing or confounding?
		1) Better interpretation of results
		2) Categories involved in collisions are not related in any meaningful way. Collisions are arbitrary so no added benefit
		3) Hashing functions have no notion of probability that each key will occur
			1) Combine Low frequency variables if possible for collisiosn so that: For example, if two categories that each account for 5% of the training set values are combined, the dummy variable has a two-fold increase in the number of ones. This would reduce the chances of being filtered out during resampling and might also lessen the potential effect that such a near-zero variance predictor might have on the analysis
	10) Common Hash Functions
		1) MurmurHash: index $=$ hash $($ feature $) \bmod d$
		2) xxHash
		3) CityHash
		4) FNV hash

## 3) Supervised encoding methods for categorical data

If there is a probability of encountering a new category in the future
- 1) Use other category
	- Can use this with Feature Hasing 
	- Need to ensure that other is present in training/testing data
	- ensure that other collides with another hashed category
	- If a specific predictor category is present in only the training set (or vice versa), a dummy variable can still be created for both data sets although it will be an zero-variance predictor in one of them.

**Supervised Encoding Methods**

Method 1) Effect or likelihood encoding
- Effect of the factor level on the outcome is measured and this effect is used as the numeric encoding
	- Numerical: Mean or Median of the outcome
	- Categorical: Logisitic Regression
		- Calculate log odds for each predictor value and use this as encoding
		- Issue: Say we have one one sample for that and it occurs: $log(1 / 1-1) =\infty$
		- So we can't trust low sample sizes
		- Solution: Shrinkage
			- Poor Quality Factor: 
				- Due to small sample size or a large variance within data for that level
			- Method: Bayesian Analysis
				- Place a prior distirbution on the results
				- Then a poor empirical with a prior results in a posterior more shifted toward mean of prior
				- Normal is a decent prior
			- Method: Empirical Bayes:
- Issues with Effect Encoding
	- Increases possiblility of overfitting: If these encodings are not consistent with future data, this will result in overfitting that the model cannot detect since it is not exposed to any other data that might contradict the findings.
		-  strongly recommended that either a different data sets be used to estimate the encodings and the predictive model or that their derivation is conducted inside resampling so that the assessment set can measure the overfitting

Method 2) Word Embeddings 
-  the idea is to estimate a smaller set of numeric features that can be used to adequately represent the categorical predictors
- Instead of representing categories with huge sparse one-hot vectors, learn compact dense numeric representations that capture similarity between categories.
	- Learn a few numeric coordinates per category
	- Embeddings turn categories into points in geometric space so similar categories end up nearby
	- Embeddings are learned during training
- The model would perform the following
	- Embeddings as inputs
	- Make predictions
	- Compute Loss
	- Update embeddings using gradient descent
- Neural Network Setting
	- Set one or more nonlinear variables between predictors and outcomes (hidden layers)
	- Use a Relu to do this: This allows the model additional complexity using nonlinear relationships.

## 4) Encodings for Ordered Data

Ordered Categories may have a linear relationship with the response or maybe even a nonlinear effect
- Need to give the model chance to uncover relationship: linear ordering
	- Known as **Polynomial contrast**

**Polynomial Contrast Embedding**
- Single comparison and its coefficients sum to zero. 
	- For low, medium, high : contrast would be -.71,0,.71
	- Quadratic then it would be .41,-.82, .41
	- Complexity: one less then number of categories
- Do this when form of the numeric representation is important
- When are these bad:
	- For example, in some cases, one might expect a trend where “low” and “middle” samples have a roughly equivalent response but “high” samples have a much different response. In this case, polynomial contrasts are unlikely to be effective at modeling this trend.
	- Moderate to high number of categories
		- Place a limit on the polynomial degree . usually quariatic
- Alternative methods
	- Treat predictors as unordered factors
	- Translate ordered categories into a single set of numeric scores based on context specific information
		- Minor fail: 1, Major fail : 10

## 5) Creating Features from Text Data

Methods
1) Different Word Analysis Approach
2) TD-IDF

**Differential Word Analysis**

Computing Odds-ratio and p value for certian workds
- Odd Ratio: How much more likely the word is in one group
- P value with FDR: expected proportion of false discoveries among declared significant results.

But issue with p value?
- p value asks: Is there a difference in the odds between the two groups? not how much
- FPR raises when using the same test on the same data so many time, so false discovery rate rises
	- Volcano Plot of odds ratio against -log (FDR)

Criteria:
- Odd ratio of at least 2 in either direction
- FDR value less than $10^-5$ 

means meaningful words and therefore features

Methods
- Baseline model
- then add features and key words in slowly

Other methods
- Removing stopping words and stemming words so that similar words such as singular and plural words are represented together
- Frequency inverse document frequency statistic
	- Find words or terms that are important to individual documents in teh collection of documents at hand
	- Potnetial issueL current collection of documents
		- This can be computed for the training set but what should be done when a single new sample is being predicted (i.e., there is no collection)? One approach is to use the overall `idf` values from the training set to weight the term frequency found in new samples.


## 6) Other

Random Notes
- Tree Based and Naive Bayes model don't require numerical features inherently while the rest do!

**Factors Versus Dummy Variables in Tree-Based Models**

Without converting to dummy variables Tree models can use this
- if day in {Sun, Sat} then ridership = 4.4K else ridership = 17.3K
- Or as dummy: if day = Sun then ridership = 3.84K else if day = Sat then ridership = 4.96K else ridership = 17.30K

Does it matter how predicitions are encoded?
- here is no real difference in the area under the ROC curve between the encoding methods.
- Ensemble methods performed better when using factors instead of dummy variables
- No real difference between two methods results wise
	- But speed favors factors rather than dummy variables by alot
- For a guideline, we suggest using the predictors without converting to dummy variables and, if the model appears promising, to also try refitting using dummy variables.
