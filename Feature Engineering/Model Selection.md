
Resampling is a good way to initally compare two different models like Neural Networks and Random Forests
- Compare Differences between perfromance across Resampling
- Use a paired t test
	- A simple approach would be to consider a paired _t_-test between the two models or an ordinary one-sample _t_-test on the differences.
	- Value in a t test approach
		- It prevents the test set from being used during the model development process
		- Many evaluations (via assessment sets) are used to gauge the differences.
			- Using multiple differences , variability in performance statistics can be measured

Bayesian Approaches Also Exist