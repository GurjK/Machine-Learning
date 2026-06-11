Classification and Regression Trees also called decision trees are defined by recursively partitioning the input space, and defining a local model in each resuling region of the input space. 

## Regression Trees

Tree is a set of decision rules
- Node $i$:
- feature dimension $d_i$
- input vector: $x$
- $threshold t_i$

At each node $i$, the feature dimension $d_i$ of the input vector $\boldsymbol{x}$ is compared to a threshold value $t_i$, and the input is then passed down to the left or right branch, depending on whether it is above or below threshold. At the leaves of the tree, the model specifies the predicted output for any input that falls into that part of the input space.

Regression Tree is defined as: $f(\boldsymbol{x} ; \boldsymbol{\theta})=\sum_{j=1}^J w_j \mathbb{I}\left(\boldsymbol{x} \in R_j\right)$
- $R_j$ is the region specified by the jth leaf node
- $w_j$ is the predicted output for that node: $w_j=\frac{\sum_{n=1}^N y_n \mathbb{I}\left(\boldsymbol{x}_n \in R_j\right)}{\sum_{n=1}^N \mathbb{I}\left(\boldsymbol{x}_n \in R_j\right)}$
- $J$ is the number of nodes
- $\boldsymbol{\theta}=\left\{\left(R_j, w_j\right): j=1: J\right\}$


First Node: $x_1<t_1$
- Yes: $x_2 < t_2$
- No $x_2< t_4$

![[Regression Tree.png]]
For example,  we have $R_1=\left[\left(x_1 \leq t_1\right),\left(x_2 \leq t_2\right)\right]$, $R_4=\left[\left(x_1 \leq t_1\right),\left(x_2>t_2\right),\left(x_3 \leq t_3\right)\right]$, etc. (For categorical inputs, we can define the splits based on comparing feature $x_i$ to each of the possible values for that feature, rather than comparing to a numeric threshold.)

**Model Fitting**

Minimize the loss: $\mathcal{L}(\boldsymbol{\theta})=\sum_{n=1}^N \ell\left(y_n, f\left(\boldsymbol{x}_n ; \boldsymbol{\theta}\right)\right)=\sum_{j=1}^J \sum_{\boldsymbol{x}_n \in R_j} \ell\left(y_n, w_j\right)$
- Not differentiable
- Greedy procedure where we grow the tree one node at a time

Method
- Suppose we are at a node $i$: $D_i$ are examples that reach this node
	- Split the node into left and right branch so as to minimize the error in each child subbranch
- Assume $j$ feature is real scalar then partition data at node $i$ based on threshold $t_i$
	- Threshold would be set to all the values that are present in the feature values
		- For example, if feature 1 has the values $\{4.5,-12,72,-12\}$, then we set $\mathcal{T}_1=\{-12,4.5,72\}$
- If $j$ feature is categorical, with $K_j$ possible values, then we check if feature is equal to each of those values or not.
- Mathematically: Now choose best feature $j_i$ and best values for that feature $t_i$ to split on: $\left(j_i, t_i\right)=\arg \min _{j \in\{1, \ldots, D\}} \min _{t \in \mathcal{T}_j} \frac{\left|\mathcal{D}_i^L(j, t)\right|}{\left|\mathcal{D}_i\right|} c\left(\mathcal{D}_i^L(j, t)\right)+\frac{\left|\mathcal{D}_i^R(j, t)\right|}{\left|\mathcal{D}_i\right|} c\left(\mathcal{D}_i^R(j, t)\right)$
- The cost function $c(\mathcal{D_i})$: $\operatorname{cost}\left(\mathcal{D}_i\right)=\frac{1}{|\mathcal{D}|} \sum_{n \in \mathcal{D}_i}\left(y_n-\bar{y}\right)^2$ (Mean Squared Error for Regression)
	- where $\bar{y}=\frac{1}{|\mathcal{D}|} \sum_{n \in \mathcal{D}} y_n$ is the mean of the response variable for examples reaching node $i$.
	- Average value in a region
- For Classification: Compute Empirical Distribution: $\hat{\pi}_{i c}=\frac{1}{\left|\mathcal{D}_i\right|} \sum_{n \in \mathcal{D}_i} \mathbb{I}\left(y_n=c\right)$
	- Then compute the Gini Index: $G_i=\sum_{c=1}^C \hat{\pi}_{i c}\left(1-\hat{\pi}_{i c}\right)=\sum_c \hat{\pi}_{i c}-\sum_c \hat{\pi}_{i c}^2=1-\sum_c \hat{\pi}_{i c}^2$
		- This is the expected error rate
		- $\hat{\pi_{ic}}$ is the probability a random entry in the leaf belong to class $c$ and $1 - \hat{\pi_{ic}}$ is probability that it would be misclassified
		- Small if one class dominates
		- large if classes are mixed
	- Majority Class in a region
	
- Cost can also be defined as entropy or deviance of the node: $H_i=\mathbb{H}\left(\hat{\boldsymbol{\pi}}_i\right)=-\sum_{c=1}^C \hat{\pi}_{i c} \log \hat{\pi}_{i c}$
	- A node that is pure will have 0 entropy
- A Differentiable decision tree or soft decisition tree, we could use a hard split diecision using a binary logistic regression model

**Regularization**

Issues with tree
- If a tree id deep enough we will get 0 training error but this results in overfitting
- Solutions
	- Stop growing tree to some heuristic, max depth or two few examples in a node
	- The second approach is to grow the tree to its maximum depth, where no more splits are possible, and then to prune it back, by merging split subtrees back into their parent
- Summary
	- Max Depth
	- Min Leaf Size
	- Minimuim Impurity Decrease
	- Cost Complexity Pruing

**Handling Missing Data**
- look for series of backup variables which can induce a similar partition to the chosen variable at any given split
	- Find highly correlated variables and just use that variable as the split
- Code missing valyes to a new values

**Pros and Cons**

Pros
- Easy to interpret
- They can easily handle mixed discrete and continuous inputs.
- They are insensitive to monotone transformations of the inputs (because the split points are based on ranking the data points), so there is no need to standardize the data
- They perform automatic variable selection
- They are relatively robust to outliers
- They are fast to fit, and scale well to large data sets
- They can handle missing input features
Cons
- that they do not predict very accurately compared to other kinds of model. due to greedy nature
- Trees are unstable
	- Small changes in input result in large effects in structure of the tree causing errors at the top to affect the rest of the tree

### Ensemble Learning

 Decision Trees are unstable and have a high variance. How do we solve this?
 - Average Multiple Models call base model or learner
	 - Different hyperparameters of the same algo
	 - different algos
	 - different data/feature set of same algo
 - Called *Ensemble Learning*
	 - Logically, say you have 3 base learners that are right 70 percent of the time
		 - Odds all 3 right? .7 * .7 * .7 = 34%
		 - Odd at least 2 right? .7 * .7 * .3 =14.7*3 = 44.1%
		 - Therfore ensemble is right at least 78% of the time!

Model: $f(y \mid \boldsymbol{x})=\frac{1}{|\mathcal{M}|} \sum_{m \in \mathcal{M}} f_m(y \mid \boldsymbol{x})$
- $f_m$ is the $m$'th base model
- Similar bias but much lower variance
- Regression: Averaging
- Classification: Majority Vote

Alternative is *stacking* : $f(y \mid \boldsymbol{x})=\sum_{m \in \mathcal{M}} w_m f_m(y \mid \boldsymbol{x})$
- Use unweighted average or majority vote to learn how to combine the base models

Ensemble is not Bayesian Model Averaging
- Ensemble: $p(y \mid \boldsymbol{x}, \boldsymbol{w}, \boldsymbol{\theta})=\sum_{m \in \mathcal{M}} w_m p\left(y \mid \boldsymbol{x}, \boldsymbol{\theta}_m\right)$
- Bayes: $p(y \mid \boldsymbol{x}, \mathcal{D})=\sum_{m \in \mathcal{M}} p(m \mid \mathcal{D}) p(y \mid \boldsymbol{x}, m, \mathcal{D})$

### Bagging

Bagging stands for "Bootstrap Aggregating". This is a simple Ensemble Method
- Fit $M$ different base models to different randomly sampled versions of the data
	- Sampled with Replacements
	- Continue until $N$ data points (Total number of data points)
- Bootstrapping is used fro sampling the training data and generating the $M$ bags
- Data can overlap between base learners
Pros
- Prevents overreliance on any one data point.
	- Advantage increases with size of ensemble
Cons
- Each model only sees 63% on average of unique data points
	- Data point will not be selected: $(1 - \frac{1}{N})^N -> .37$
		- $N \cdot\left(1-\left(1-\frac{1}{N}\right)^N\right) \sim N(1-\exp (-1)) \sim 0.632 \cdot N$
	- 37% are called out of bag instances. Can be used as test set
- Bagging does not always improve performance. In particular, it relies on the base models being unstable estimators, so that omitting some of the data significantly changes the resulting model fit
- Base Learns may overfit the data and have low bias and high variance

Steps
- Generate $M$ bags from training data $D$ by sampling with replacement
	- Same number of features but only a certain number of data points
- Generate base Learners and train their bags on them
- Final Estimate is average or max vote across $M$ models
### Boosting

Bagging Models: $f(\boldsymbol{x} ; \boldsymbol{\theta})=\sum_{m=1}^M \beta_m F_m\left(\boldsymbol{x} ; \boldsymbol{\theta}_m\right)$
- $F_m$ is the mth tree
- $\beta_m$ is the corresponding weight
	- $\beta_m = 1/M$
- Result is an additive model
	- linear model with adaptive basis functions
	- Empirical loss with an optional regularizer: $\mathcal{L}(f)=\sum_{i=1}^N \ell\left(y_i, f\left(\boldsymbol{x}_i\right)\right)$

Boosting is an algorithm for sequentially fitting additive models where each $F_m$ is a binary classifier that returns $F_m \in (-1,+1)$ 
- $F_1$ is fit on original data
	- Weight data samples by the errors made by $F_1$ so misclassified examples get more weight
- $F_2$ to this weighted data set
	- Keep repeating until we have $M$ of components
	- $M$ is hyperparameter that controls complexity of overall model and can be chosen by monitoring performance on validation set and using early stopping
- Start with a weak learner and end with a strong learner
	- Boosting reduces the bias of the strong learning by fitting trees that depend on each other
	- Boosting gradually reduces systematic error, which improves approximation accuracy
		- Bias is cause by a too simple model so predictions are wrong in a 

*It can be shown that, as long as each Fm has an accuracy that is better than chance (even on the weighted dataset), then the final ensemble of classifiers will have higher accuracy than any given component.*

**Forward Stagewise additive modelling**

Sequentially optimize the empirical loss for general (differentiable) loss functions, where $f$ is an additive model: At iteration $m$ we compute: $$
\left(\beta_m, \boldsymbol{\theta}_m\right)=\underset{\beta, \boldsymbol{\theta}}{\operatorname{argmin}} \sum_{i=1}^N \ell\left(y_i, f_{m-1}\left(\boldsymbol{x}_i\right)+\beta F\left(\boldsymbol{x}_i ; \boldsymbol{\theta}\right)\right)
$$
- $f_{m-1}$: Current model before iteration $m$
	- $f_{m-1}(x)=\sum_{k=1}^{m-1} \beta_k F\left(x ; \theta_k\right)$
- $F(x_i;\theta)$: new weak learner candidate
- $\beta$: Step size or weight of the new learner
- $y_i$: True target value for observation $i$

Then the model updates to $$
f_m(\boldsymbol{x})=f_{m-1}(\boldsymbol{x})+\beta_m F\left(\boldsymbol{x} ; \boldsymbol{\theta}_m\right)=f_{m-1}(\boldsymbol{x})+\beta_m F_m(\boldsymbol{x})
$$

Suppose we used squared error loss: $\ell(y, \hat{y})=(y-\hat{y})^2$, Then we would get the following for the $ith$ term at step $m$: $$
\ell\left(y_i, f_{m-1}\left(\boldsymbol{x}_i\right)+\beta F\left(\boldsymbol{x}_i ; \boldsymbol{\theta}\right)\right)=\left(y_i-f_{m-1}\left(\boldsymbol{x}_i\right)-\beta F\left(\boldsymbol{x}_i ; \boldsymbol{\theta}\right)\right)^2=\left(r_{i m}-\beta F\left(\boldsymbol{x}_i ; \boldsymbol{\theta}\right)\right)^2
$$
- $r_{i m}=y_i-f_{m-1}\left(\boldsymbol{x}_i\right)$ is the residual of the current model on the $i$ 'th observation.
	- residual of the current model on  the ith observation
	- Can minimize the above by setting $\beta =1$ and fitting $F$ to the residuals
	- This is called **Least Sqaures Boosting**
		- For squared loss: Gradient equals the residuals: $-\frac{\partial \ell}{\partial f}=y-f(x)$
	- 


**Feature Importance for Trees**

For a single tree $T$ , the feature importance of feature $k$ is $$
R_k(T)=\sum_{j=1}^{J-1} G_j \mathbb{I}\left(v_j=k\right)
$$
- sum is over all non-leaf (internal) nodes
- $G_j$ is the gain in accuracy at node $j$, and $v_j =k$ if node $j$ uses feature $k$
- Can get a more reliable estimate by averaging over all trees in the ensemble: $$
R_k=\frac{1}{M} \sum_{m=1}^M R_k\left(T_m\right)
$$
- Normalize them so largest is 100%

*Partial Depedency Plots*

We can access the impact features have on output using a partial dependency plot: For feature $k$ this is $$
\bar{f}_k\left(x_k\right)=\frac{1}{N} \sum_{n=1}^N f\left(\boldsymbol{x}_{n,-k}, x_k\right)
$$
- vs $x_k$
- Then marginalize out all features except $k$
	- BC: Convert to log odds:  $\log p\left(y=1 \mid x_k\right) / p\left(y=0 \mid x_k\right)$, before plotting

