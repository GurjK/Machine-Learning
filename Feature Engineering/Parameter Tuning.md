Hyperparameters are specific to the model and control the complexity of the model. To make sure that proper values of the tuning parameters are used, some sort of search procedure is required along with a method for obtaining good, generalizable measures of performance. For the latter, repeatedly using the test set for these questions is problematic since it would lose its impartiality. Instead, resampling is commonly used.

2 Main Methods Exists
- 1) Grid Search: a set of candidate tuning parameter values are specified and then evaluated.
	- We recommend using resampling to evaluate each distinct parameter value combination to get good estimates of how well each candidate performs. Once the results are calculated, the “best” tuning parameter combination is chosen and the final model is fit to the entire training set with this value.
	- Pick combination with best empirical results
- 2) Random Search Grid: to define a range of possible values for each parameter and to randomly sample the multidimensional space enough times to cover a reasonable amount
	- When there is no notion of what the values should be !
	- Neural Network, GBM and other models can be effectively tuned using approach
- Nelder Mead Simplex
	- simulated annealing and genetic algos
	- One reason for this is that each evaluation of a new parameter requires a good estimate of performance to guide the search. Resampling is one of the best methods for doing this.
- Bayesian Optimization
	- Creates a separate model to predict performance as a function of tuning parameters and then make a recommendation as to the next candidate set to evaluate
	- An inital pool of samples are evaluated using grid or random search here

By choosing the best setting we are introducing optimization bias
-  Depending on the problem, this bias might overestimate the model’s true performance. There are nested resampling procedures that can be used to mitigate these biases.