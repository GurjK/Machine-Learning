
Overview
- 1) Training and Test Data
- 2) Resampling
	- v-Fold Cross Validation
	- Monte Carlo Cross Validation
	- Bootstrap



#### 1) Training and Test Data

Common to split the data into training and testing set
- 1) Some version of random sampling
	- Completely random sampling prevents bias toward any characteristic
	- Proble: Response is not evenly distributed
- 2) Stratified random sample based on outcome
	- Classification: Samples at random within class
- 3) Non Random
	- A temporal aspect to the data exists

Validation Sets
- In between Test and training data
- Often, the validation set is a random subset of the training set so that it changes for each iteration of the numerical optimizer. In resampling this type of model, it would be reasonable to use a small, random portion of the analysis set to serve as a within-resample validation set.
#### 2) Resampling

Resampling methods can generate different versions of our training set that can be used to simulate how well models would perform on new data.
- Generates a subset of the data used for modeling and another that is uses for measuring performance
	- Analysis set and assesment set


![[Data Splitting.png]]

##### K Fold Cross Validation

Creates $K$ different versions of original training set that have same approximate size
- Each $K$ contains: $1/K$ of the training set and each excludes different point
- The analysis sets contain the remainder (typically called the “folds”) . Suppose _K_ = 10, then there are 10 different versions of 90% of the data and also 10 versions of the remaining 10% for each corresponding resample.
- Model is created on first fold and corrspeonding assesment set if predicted by the model
	- summarized using metric like RMSE
- In the end we have $K$ different estimations of the model
- Average the $K$ metrics

Simple example (K = 5)

You split your data into 5 folds:

- Run 1: Train on folds 2–5, test on fold 1
- Run 2: Train on folds 1,3,4,5, test on fold 2
- Run 3: Train on folds 1,2,4,5, test on fold 3
- Run 4: Train on folds 1–3,5, test on fold 4
- Run 5: Train on folds 1–4, test on fold 5

Then you average the 5 test scores.

If outcome is Categorical use Stratified splitting techniques are used to produce same frequency distribution of the outcome
- Again, this is a good idea when a continuous outcome is skewed or a categorical outcome is imbalanced, but is unlikely to be problematic otherwise
- Each fold preserves the class distribution

Downsides
- Relative noiser (more variabilitiy)
	- Solution: Conduct repeated K fold cross validation
	- if R repreats are used with K resamples are recreated R separate times then RK samples are used 
		- reduced the variance by $\sqrt{R}$

Other Variations
- Leave one out cross validation
	- $K$ is equal to size of training set
- Grouped $K$ fold cross validation
- Leave Group Out cross validation

##### Monte Carlo Cross Validation

Monte Carlo resampling produces splits that are likely to contain overlaps
- A random sample sample is take with $\pi$ proportion of the training set going into analysis set and reamining samples allocated to assesment
- model is created on the analysis set and assesment to evaluate model
- Splitting feature is conduced $B$ times and average of $B$ are used to estimate future performance

Summary
- Same as $K$ fold CV but testing set across the splits can contain some the same data points

##### Bootstrap

A bootstrap resample of the data is defined to be a simple random sample that is the same size as the training set where the data are sampled _with replacement_
- Bootstrap resample is created there is a 63.2% chance that any training set memeber is included in the bootstrap sample at least once
- The bootstrap resample is used as the analysis set and the assessment set, sometimes known as the out-of-bag sample, consists of the members of the training set not included in the bootstrap sample.
	- bootstrap sampling is conducted _B_ times and the same modeling/evaluation procedure is followed to produce a bootstrap estimate of performance that is the mean of _B_ results.


| Method | Data usage | Overlap in test sets | Stability | Typical use |
| :--- | :--- | :--- | :--- | :--- |
| K-fold | Structured partitions | No | High | Standard evaluation |
| Monte Carlo CV | Random splits | Yes | Medium-High | Flexible evaluation |
| Bootstrapping | Sampling w/ replacement | Yes (OOB testing) | Medium | Uncertainty, small data |
##### Rolling Origin Forecasting

If there are seasonal or other chronic trends in the data, random splitting of data between the analysis and assessment sets may disrupt the model’s ability to estimate these patterns
- First Analsis: $M$ training set points
	- Training set is ordered by time
- Assesment: Next $N$ points
- Next run would be points M+1
	- Assesment : Moints M+2 to M+N+1

Walk forward Cross Validation
-  However, it should be understood that, since there is significant overlap in the rolling assessment sets, the _B_ samples themselves constitute a time-series and might also display seasonal or other temporal effects.

Variations
- Growing/Expanding Window
- Rolling Origin Evaluation
- Rolling Winodw/Fixed Window
- Skipped/Blocked Time series CV
- Irregular Time spacing

###### Variance and Bias in Resampling

Variance: measured by determining spread of the resulting averages in CV
Bias: ability of a particular resampling scehme to be able to hit the true underlying performance parameter
- Bias in 10 fold is smaller than in 5 fold 
- 10 fold has less bias then MC CV when $\pi=10$ is used
- Leave one out has low bias

![[Variance and Bias in Resampling.png]]

Simple CV: Noisiest(High Variance)
- Bias is low when V is 10 and repeat 5 or so times when low training set
Bootstrap : least variable, bias means likely to show worse model performance than true underlying value


As a part of Resampling must include
- PCA , Feature Selection methods
- Missing data transformations
- Center and Scaling can be exempted from resampling tho
- Downsampling

