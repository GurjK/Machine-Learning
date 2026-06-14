
$$
\begin{array}{lll}
\text { Name } & \text { Loss } & -\partial \ell\left(y_i, f\left(\boldsymbol{x}_i\right)\right) / \partial f\left(\boldsymbol{x}_i\right) \\
\hline \text { Squared error } & \frac{1}{2}\left(y_i-f\left(\boldsymbol{x}_i\right)\right)^2 & y_i-f\left(\boldsymbol{x}_i\right) \\
\text { Absolute error } & \left|y_i-f\left(\boldsymbol{x}_i\right)\right| & \operatorname{sgn}\left(y_i-f\left(\boldsymbol{x}_i\right)\right) \\
\text { Exponential loss } & \exp \left(-\tilde{y}_i f\left(\boldsymbol{x}_i\right)\right) & -\tilde{y}_i \exp \left(-\tilde{y}_i f\left(\boldsymbol{x}_i\right)\right) \\
\text { Binary Logloss } & \log \left(1+e^{-\tilde{y}_i f_i}\right) & y_i-\pi_i \\
\text { Multiclass logloss } & -\sum_c y_{i c} \log \pi_{i c} & y_{i c}-\pi_{i c}
\end{array}
$$


General version of boosting know as gradient boosting
- at each step $m$, Let $g_m$ be the gradient of $\mathcal{L(f)}$ evaluated at $f = f_{m-1}$:
	- $g_{i m}=\left[\frac{\partial \ell\left(y_i, f\left(\boldsymbol{x}_i\right)\right)}{\partial f\left(\boldsymbol{x}_i\right)}\right]_{f=f_{m-1}}$
	- Key is that here we have a prediction function$f(x)$ and evaluating it on all the training points, getting the vector $f$
	- Compute the gradient of the loss with respect to this function's output not its parameters
		- This is $g_m$: s a vector pointing in the direction of steepest _increase_ in loss for each training point
- Update them $f_m = f_{m-1} - \beta_mg_m$
- where $\beta_m$ is the step length chosen by : $\beta_m=\underset{\beta}{\operatorname{argmin}} \mathcal{L}\left(\boldsymbol{f}_{m-1}-\beta \boldsymbol{g}_m\right)$
	- $-\beta_mg_m$ is just a set of values defined only at the training points. Can't use it at new $x$ values. 
	- This is done by line search. In modern implementations with tree-based weak learners, this line search is often skipped because a separate optimal ββ (weight) is found for each _terminal node_ of the tree, which is more powerful.
		- $f_m(x)=f_{m-1}(x)+\nu \sum_j \gamma_{j m} 1\left(x \in R_{j m}\right)$, where $\gamma_{j m}$ is the optimal update for leaf $R_{j m}$.
- Fit a weak learner to approximate the negative gradient signal
- The implementable update is then: $f_m(x)=f_{m-1}(x)+\nu \cdot h_m(x)$

Then update is then: $$
F_m=\underset{F}{\operatorname{argmin}} \sum_{i=1}^N\left(-g_{i m}-F\left(\boldsymbol{x}_i\right)\right)^2
$$
- shrinkage factor $0<\nu \leq 1$, to control the size of the updates, for regularization purposes.
	- A smaller ν (e.g., 0.01) almost always requires a larger number of trees MM but leads to significantly better generalization.

If applying this using squared loss we get L2Boosting, $-g_{im} = y_i -f_{m-1}(x_i)$
- residual error
For Classification
- Use log-loss. Then we get BinomialBoost
- Advtange of LogitBoost is that it does not need to be able to do weighted fitting:
	- Applies any black box regression model to the gradient vector
	- To apply to multi-class classifcation: fit $C$ separate regression trees, using pseudo residual of the form: $$
-g_{i c m}=\frac{\partial \ell\left(y_i, f_{1 m}\left(\boldsymbol{x}_i\right), \ldots, f_{C m}\left(\boldsymbol{x}_i\right)\right)}{\partial f_{c m}\left(\boldsymbol{x}_i\right)}=\mathbb{I}\left(y_i=c\right)-\pi_{i c}
$$
- Although the trees are fit separately, their predictions are combined via a softmax transformation: $$
p(y=c \mid \boldsymbol{x})=\frac{e^{f_c(\boldsymbol{x})}}{\sum_{c^{\prime}=1}^C e^{f_{c^{\prime}}(\boldsymbol{x})}}
$$
- Stochastic setting we can subsample a random fraction of the data to pass to the regression tree at each iteration. Called Stochastic Gradient Boosting
	- Generalizes bettter because subsampling is a form of regualrization
- Regularization
	- Number of trees: $M$
	- Shrinkage
	- SGB
	- Feature Bagging
	- Tree Constraints
		- Max Depth
		- Min Samples/Splits
		- Min Loss Reduction

**Gradient Tree Boosting**

Gradient Boosting is almost always implements with a tree as the weak learner:

Model: $F_m(\boldsymbol{x})=\sum_{j=1}^{J_m} w_{j m} \mathbb{I}\left(\boldsymbol{x} \in R_{j m}\right)$
- $w_{jm}$ is predicted output for region $R_{jm}$
- 

#### Gradient Boost


**Loss Functions**

Squared Error: L2Boosting
- Mean Regression: $E[Y|X]$
- Weak to outlier
Absolute Error: L1Boosting
- Median Regression: $Median[Y|X]$
Quantile Loss:
- Predice 95th percentile not just mean
Huber Loss:
- Quadratic for small residuals and absolute for large