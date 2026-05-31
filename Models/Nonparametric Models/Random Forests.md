Regular Bagging
- Re-Running the same learning algorithm on different subsets of data will result in sufficient diverse base model

Random Forest
- Tries to decorrelate base trees by using
	- Random Features
	- Random Data

How?
- Feature split dimension $j$ is optimized over a random subset of the features: $S_i \subset\{1, \ldots, D\}$.
- RF is better than dcisition trees because many input features are irrelevant


