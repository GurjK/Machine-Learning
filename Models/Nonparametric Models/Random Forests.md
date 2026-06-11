Regular Bagging
- Re-Running the same learning algorithm on different subsets of data will result in sufficient diverse base model

Limitations of Bagging
- Learning limited due to "out of bag" common across many trees
- Deep over fitted trees lead of odd fluctuations (leaf count too small)
- Data correlation across trees limit "Variance reduction" from bagging

$$ \sigma^2 / T \to \rho \sigma^2 + \frac{1-\rho}{T} \sigma^2$$

Random Forest
- Tries to decorrelate base trees by using
	- Random Features
	- Random Data
- Random subspace method for building trees
- Validate out of bag errors during training
- Ability to tune hyper parameters for underlying algorithm

How?
- Feature split dimension $j$ is optimized over a random subset of the features: $S_i \subset\{1, \ldots, D\}$.
- RF is better than dcisition trees because many input features are irrelevant


