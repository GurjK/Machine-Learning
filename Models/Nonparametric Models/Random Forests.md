Regular Bagging
- Re-Running the same learning algorithm on different subsets of data will result in sufficient diverse base model

Limitations of Bagging
- Learning limited due to "out of bag" common across many trees
	- 63% probability for a data point to be out of a bag
- Deep over fitted trees lead of odd fluctuations (leaf count too small)
	- No regularization or optimization: leaf count too low etc
- Data correlation across trees limit "Variance reduction" from bagging

$$ \sigma^2 / T \to \rho \sigma^2 + \frac{1-\rho}{T} \sigma^2$$

Random Forest
- Tries to decorrelate base trees by using
	- Random Features
	- Random Data
- Random subspace method for building trees
	- Sampling features
- Validate out of bag errors during training
	- cross validation on out of bag data points
- Ability to tune hyper parameters for underlying algorithm
	- Regularization can be imposed

How?
- Feature split dimension $j$ is optimized over a random subset of the features: $S_i \subset\{1, \ldots, D\}$.
- RF is better than dcisition trees because many input features are irrelevant


Steps:
- Given Training Data $D$, Generate $M$ bags from training data by sampling with replacement. Both features and data points are sampled
- 