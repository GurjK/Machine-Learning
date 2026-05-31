Maximum likelihood estimation can result in overfitting on plain OLS. We may not fit well on new future data leading to **overfitting**. The core problem is that the model has enough parameters to perfectly fit the training data so it can perfectly match the empirical distribution. The empirical distribution may not be the same as the true distribution therefore the model may not generalize well.

The solution to this is to use **regularization** , so adding a penalty term to the NLL or empirical risk as follows. 

$$\mathcal{L}(\boldsymbol{\theta} ; \lambda)=\left[\frac{1}{N} \sum_{n=1}^N \ell\left(\boldsymbol{y}_n, \boldsymbol{\theta} ; \boldsymbol{x}_n\right)\right]+\lambda C(\boldsymbol{\theta})$$
- $\lambda \geq 0$: Regularization parameter
- $C(\theta)$: Complex penalty

A common complexity penalty is to use $C(\boldsymbol{\theta})=-\log p(\boldsymbol{\theta})$, where $p(\boldsymbol{\theta})$ is the prior for $\boldsymbol{\theta}$. If $\ell$ is the log loss, the regularized objective becomes

$$
\begin{equation*}
\mathcal{L}(\boldsymbol{\theta} ; \lambda)=-\frac{1}{N} \sum_{n=1}^N \log p\left(\boldsymbol{y}_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}\right)-\lambda \log p(\boldsymbol{\theta}) 
\end{equation*}
$$

By setting $\lambda=1$ and rescaling $p(\boldsymbol{\theta})$ appropriately, we can equivalently minimize the following

$$
\mathcal{L}(\boldsymbol{\theta} ; \lambda)=-\left[\sum_{n=1}^N \log p\left(\boldsymbol{y}_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}\right)+\log p(\boldsymbol{\theta})\right]=-[\log p(\mathcal{D} \mid \boldsymbol{\theta})+\log p(\boldsymbol{\theta})]
$$


Minimizing this is equivalent to maximizing the log posterior:

$$
\hat{\boldsymbol{\theta}}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \log p(\boldsymbol{\theta} \mid \mathcal{D})=\underset{\boldsymbol{\theta}}{\operatorname{argmax}}[\log p(\mathcal{D} \mid \boldsymbol{\theta})+\log p(\boldsymbol{\theta})-\text { const }]
$$


This is known as MAP estimation, which stands for maximum a posterior estimation.




Suppose we use map estimation with a zero mean Gaussian prior on the weights where 

$p(\boldsymbol{w})=\mathcal{N}\left(\boldsymbol{w} \mid \mathbf{0}, \lambda^{-1} \mathbf{I}\right)$. This is called ridge regression.
$$
\begin{equation*}
\hat{\boldsymbol{w}}_{\text {map }}=\underset{\boldsymbol{w}}{\operatorname{argmin}} \mathrm{NLL}(\boldsymbol{w})+\lambda\|\boldsymbol{w}\|_2^2 
\end{equation*}
$$

where $\|\boldsymbol{w}\|_2^2=\sum_{d=1}^D w_d^2$. (We write $\boldsymbol{w}$ rather than $\boldsymbol{\theta}$, since it only really make sense to penalize the magnitude of weight vectors, rather than other parameters, such as bias terms or noise variances.)

This is called $\ell_2$ regularization or weight decay. The larger the value of $\lambda$, the more the parameters are penalized for being "large" (deviating from the zero-mean prior), and thus the less flexible the model.

The MAP estiamte is computed as follows: 
$$
\begin{aligned}
\hat{\boldsymbol{w}}_{\text {map }} & =\operatorname{argmin} \frac{1}{2 \sigma^2}(\boldsymbol{y}-\mathbf{X} \boldsymbol{w})^{\top}(\boldsymbol{y}-\mathbf{X} \boldsymbol{w})+\frac{1}{2 \tau^2} \boldsymbol{w}^{\top} \boldsymbol{w} \\
& =\operatorname{argmin} \operatorname{RSS}(\boldsymbol{w})+\lambda\|\boldsymbol{w}\|_2^2
\end{aligned}
$$

where $\lambda \triangleq \frac{\sigma^2}{\tau^2}$ is proportional to the strength of the prior, and

$$
\|\boldsymbol{w}\|_2 \triangleq \sqrt{\sum_{d=1}^D\left|w_d\right|^2}=\sqrt{\boldsymbol{w}^{\top} \boldsymbol{w}}
$$
- $l_2$ normof the vector $w$
- Do not penalize the offset term (intercept)

#### Computing the MAP Estimate

Map Estimate corresponds to the following minimization problem: 

$J(\boldsymbol{w})=(\boldsymbol{y}-\mathbf{X} \boldsymbol{w})^{\boldsymbol{\top}}(\boldsymbol{y}-\mathbf{X} \boldsymbol{w})+\lambda\|\boldsymbol{w}\|_2^2$
- $\lambda = \sigma^2 / \tau^2$

Setting this equal to 0 after taking the derivative gives: 
$\nabla_{\boldsymbol{w}} J(\boldsymbol{w})=2\left(\mathbf{X}^{\top} \mathbf{X} \boldsymbol{w}-\mathbf{X}^{\top} \boldsymbol{y}+\lambda \boldsymbol{w}\right)$

Therefore the MAP estimate of this is 
$\hat{\boldsymbol{w}}_{\text {map }}=\left(\mathbf{X}^{\top} \mathbf{X}+\lambda \mathbf{I}_D\right)^{-1} \mathbf{X}^{\top} \boldsymbol{y}=\left(\sum_n \boldsymbol{x}_n \boldsymbol{x}_n^{\top}+\lambda \mathbf{I}_D\right)^{-1}\left(\sum_n y_n \boldsymbol{x}_n\right)$
- Note that $w^T \Lambda w = w^T \lambda I w = \lambda ||w||^2$
- 

Methods to Solve this Numerically?
- QR Decomposition
	- Create Extra Fakse observations with a prior of $p(\boldsymbol{w})=\mathcal{N}\left(\mathbf{0}, \boldsymbol{\Lambda}^{-1}\right), \text { where } \boldsymbol{\Lambda} \text { is the precision matrix }$
	- Then Data is $\tilde{\mathbf{X}}=\binom{\mathbf{X} / \sigma}{\sqrt{\boldsymbol{\Lambda}}}, \quad \tilde{\boldsymbol{y}}=\binom{\boldsymbol{y} / \sigma}{\mathbf{0}_{D \times 1}}$
	- Then the map estimate is: $\hat{\boldsymbol{w}}_{\operatorname{map}}=\left(\tilde{\mathbf{X}}^{\top} \tilde{\mathbf{X}}\right)^{-1} \tilde{\mathbf{X}}^{\top} \tilde{\boldsymbol{y}}$ so solve with normal OLS Estimates
- SVD
	- Faster than QR Decomposition if $D>N$
	- $X=USV^T$, $V^TV=I$, $U^TU,UU^T=I$, $S=Diag$
	- $R=US$
	- $\hat{\boldsymbol{w}}_{\text {map }}=\mathbf{V}\left(\mathbf{R}^{\top} \mathbf{R}+\lambda \mathbf{I}_N\right)^{-1} \mathbf{R}^{\top} \boldsymbol{y}$
- Ridge and PCA
	- $\hat{w}_{\text {ridge }}=\left(X^{\top} X+\lambda I\right)^{-1} X^{\top} y$ now use the SVD solution of this $\hat{\boldsymbol{w}}_{\text {map }}=\mathbf{V}\left(\mathbf{R}^{\top} \mathbf{R}+\lambda \mathbf{I}_N\right)^{-1} \mathbf{R}^{\top} \boldsymbol{y}$
		- $\hat{\boldsymbol{y}}=\mathbf{X} \hat{\boldsymbol{w}}_{\text {map }}=\sum_{j=1}^D \boldsymbol{u}_j \frac{\sigma_j^2}{\sigma_j^2+\lambda} \boldsymbol{u}_j^{\top} \boldsymbol{y}$
		- Here $\sigma$ are the singular values
	- OLS scales: $1/ \sigma$ but ridge uses $\sigma^2 / \sigma^2 + \lambda$
	- If $\sigma_i$ is small then OLS would blow it up but Ridge would supress it
		- If $\sigma_j^2$ is large compared to $\lambda$, then $\sigma_j^2 /\left(\sigma_j^2+\lambda\right) \approx \sigma_j^2 / \sigma_j^2=1$, so direction $\boldsymbol{u}_j$ is not affected, but if $\sigma_j^2$ is small compared to $\lambda$, and if $\lambda$ is large, then $\sigma_j^2 /\left(\sigma_j^2+\lambda\right) \approx 1 / \lambda \approx 0$, so direction $\boldsymbol{u}_j$ will be downweighted. 
		- $\text { When } \lambda=0, \operatorname{dof}(\lambda)=D, \text { and as } \lambda \rightarrow \infty, \operatorname{dof}(\lambda) \rightarrow 0$
	- Direction that we are most uncertain about $w$ are the largest eigenvector of $(X^TX)^{-1}$
		- Same direction as $X^TX$ with smallest eigenvalues, so also smallest singular values
		- If $\sigma_j^2$ is small relative to $\lambda$ then ridge will downwieght direction $u_j$
		- In this way, ill-determined parameters are reduced in size towards 0. called **shrinkage**
	- Principal Component Regression
		- The idea is this: first use PCA to reduce the dimensionality to K dimensions, and then use these low dimensional features as input to regression. However, this technique does not work as well as ridge regression in terms of predictive accuracy . The reason is that in PC regression, only the first K (derived) dimensions are retained, and the remaining D − K dimensions are entirely ignored. By contrast, ridge regression uses a “soft” weighting of all the dimensions.
### How to choose $\lambda$

- Small means potentially overfitting
- Large means potentially underfitting

Therefore, try finite number of Distinct values and use cross validation to estimate their expected loss. 
- If expensive start with a strong regularization and then slowly relax the constraints. Then we get a Regularization path

Bayes Approach
- $\text { computing } \lambda=\operatorname{argmax}_\lambda \log p(\mathcal{D} \mid \lambda) \text {, where } p(\mathcal{D} \mid \lambda) \text { is the marginal likelihood or evidence. }$
- Usually same result as CV
	- Bayesian approach has several advantages: computing $p(\mathcal{D} \mid \lambda)$ can be done by fitting a single model, whereas CV has to fit the same model $K$ times; and $p(\mathcal{D} \mid \lambda)$ is a smooth function of $\lambda$, so we can use gradient-based optimization instead of discrete search.

Cross Validation Approach
Split the data into training and validation sets.
- Fit model onto $D_{Train}$,.Evaluate performance on $D_{valid}$
- Pick value of $\lambda$ that results in best validation performance

$$
R_\lambda(\boldsymbol{\theta}, \mathcal{D})=\frac{1}{|\mathcal{D}|} \sum_{(\boldsymbol{x}, \boldsymbol{y}) \in \mathcal{D}} \ell(\boldsymbol{y}, f(\boldsymbol{x} ; \boldsymbol{\theta}))+\lambda C(\boldsymbol{\theta})
$$


For each $\lambda$, we compute the parameter estimate

$$
\hat{\boldsymbol{\theta}}_\lambda\left(\mathcal{D}_{\text {train }}\right)=\underset{\boldsymbol{\theta}}{\operatorname{argmin}} R_\lambda\left(\boldsymbol{\theta}, \mathcal{D}_{\text {train }}\right)
$$


We then compute the validation risk:

$$
R_\lambda^{\mathrm{val}} \triangleq R_0\left(\hat{\boldsymbol{\theta}}_\lambda\left(\mathcal{D}_{\mathrm{train}}\right), \mathcal{D}_{\mathrm{valid}}\right)
$$


This is an estimate of the population risk, which is the expected loss under the true distril $p^*(\boldsymbol{x}, \boldsymbol{y})$. Finally we pick

$$
\lambda^*=\underset{\lambda \in \mathcal{S}}{\operatorname{argmin}} R_\lambda^{\mathrm{val}}
$$

(This requires fitting the model once for each value of $\lambda$ in $\mathcal{S}$, although in some cases, this done more efficiently.)

After picking $\lambda^*$, we can refit the model to the entire dataset, $\mathcal{D}=\mathcal{D}_{\text {train }} \cup \mathcal{D}_{\text {valid }}$, to get

$$
\hat{\boldsymbol{\theta}}^*=\underset{\boldsymbol{\theta}}{\operatorname{argmin}} R_{\lambda^*}(\boldsymbol{\theta}, \mathcal{D})
$$
Issues
- Small Dataset can result in unreliable estimate of model parameters
- Instead use Cross Validation
- Plot $\lambda$ vs the mean squared error to determine this!

### Cross Validation

Split dataset into $K$ folds, train on $K-1$ folds and test on the $Kth$ fold
- $R_\lambda^{\mathrm{cv}} \triangleq \frac{1}{K} \sum_{k=1}^K R_0\left(\hat{\boldsymbol{\theta}}_\lambda\left(\mathcal{D}_{-k}\right), \mathcal{D}_k\right)$

We can use the CV estimate as an objective inside of an optimization routine to pick the optimal hyperparameter, $\hat{\lambda}=\operatorname{argmin}_\lambda R_\lambda^{\text {cv }}$. Finally we combine all the available data (training and validation), and re-estimate the model parameters using $\hat{\boldsymbol{\theta}}=\operatorname{argmin}_{\boldsymbol{\theta}} R_{\hat{\lambda}}(\boldsymbol{\theta}, \mathcal{D})$. 
- Pick simplest model whose risk is no more than one standard error above the risk of the best model. Known as the **One Standard Error Rule**

### Map Estimator for Linear Regression

Assume a gaussian prior, $p(\boldsymbol{w})=\mathcal{N}\left(\boldsymbol{w} \mid \mathbf{0}, \lambda^{-1} \mathbf{I}\right)$. 
- $\lambda$ controlls the strength of the prior
	- $\lambda = 0$ -> MLE
	- $\lambda >0$ -> Biased
		- Lightly regularized results in high variance ( spread training fits) but low bias (close to truth)
		- Highly regularized results in low variance (similar training fits) but high bias (far from truth)
- 0 mean encourages the wieghts to be small

![[Bias Variance Tradeoff.png|697]]