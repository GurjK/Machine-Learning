
In frequentist statistics, uncertainty is not represented by the posterior distribution of a random variable, but instead by the sampling distribution of an estimator.


The key idea in frequentist statistics is to view the data $\mathcal{D}$ as a random variable, and the parameters from which the data are drawn, $\boldsymbol{\theta}^*$, as a fixed but unknown constant. Thus $\hat{\boldsymbol{\theta}}=\hat{\Theta}(\mathcal{D})$ is a random variable, and its distribution is known as the sampling distribution of the estimator. To understand what thus means, suppose we create $S$ different datasets, each of the form

$$
\begin{equation*}
\mathcal{D}^{(s)}=\left\{\boldsymbol{x}_n \sim p\left(\boldsymbol{x}_n \mid \boldsymbol{\theta}^*\right): n=1: N\right\} 
\end{equation*}
$$

Sampling Distribution can be approximated as 
- Gaussian
- Monte Carlo

### Gaussian Approximation of Sampling Distribution of the MLE

Theorem:
If the parameters are identifiable, then

$$
\begin{equation*}
\operatorname{SamplingDist}\left(\hat{\Theta}^{\mathrm{mle}}, \boldsymbol{\theta}^*\right) \rightarrow \mathcal{N}\left(\cdot \mid \boldsymbol{\theta}^*,\left(N \mathbf{F}\left(\boldsymbol{\theta}^*\right)\right)^{-1}\right) 
\end{equation*}
$$

where $\mathbf{F}\left(\boldsymbol{\theta}^*\right)$ is the $\mathbf{F i s h e r}$ information matrix, defined in Equation (4.220).
Equivalently, the above result says that the distribution of $\sqrt{N F\left(\boldsymbol{\theta}^*\right)}\left(\hat{\boldsymbol{\theta}}-\boldsymbol{\theta}^*\right)$ approaches $\mathcal{N}(\mathbf{0}, \mathbf{I})$, where $\hat{\boldsymbol{\theta}}=\hat{\Theta}^{\text {mle }}(\tilde{\mathcal{D}})$.

The Fisher information matrix measures the amount of curvature of the log-likelihood surface at its peak, as we show below. More formally, the Fisher information matrix (FIM) is defined to be the covariance of the gradient of the log likelihood (also called the score function):

$$
\begin{equation*}
\mathbf{F}(\boldsymbol{\theta}) \triangleq \mathbb{E}_{\boldsymbol{x} \sim p(\boldsymbol{x} \mid \boldsymbol{\theta})}\left[\nabla \log p(\boldsymbol{x} \mid \boldsymbol{\theta}) \nabla \log p(\boldsymbol{x} \mid \boldsymbol{\theta})^{\top}\right] 
\end{equation*}
$$


Hence the $(i, j)^{\prime}$ 'th entry has the form

$$
\begin{equation*}
F_{i j}=\mathbb{E}_{\boldsymbol{x} \sim \boldsymbol{\theta}}\left[\left(\frac{\partial}{\partial \theta_i} \log p(\boldsymbol{x} \mid \boldsymbol{\theta})\right)\left(\frac{\partial}{\partial \theta_j} \log p(\boldsymbol{x} \mid \boldsymbol{\theta})\right)\right] 
\end{equation*}
$$


One can show the following result.

Theorem 
If $\log p(\boldsymbol{x} \mid \boldsymbol{\theta})$ is twice differentiable, and under certain regularity conditions, the FIM is equal to the expected Hessian of the NLL, i.e.,

$$
\begin{equation*}
\mathbf{F}_{i j}=-\mathbb{E}_{\boldsymbol{x} \sim \boldsymbol{\theta}}\left[\frac{\partial^2}{\partial \theta_i \theta_j} \log p(\boldsymbol{x} \mid \boldsymbol{\theta})\right] 
\end{equation*}
$$


If we replace the expectation over $\boldsymbol{x}$ with the observed value, to get the empirical FIM, we see this is equal to the Hessian of the NLL. This helps us understand the result in Equation (4.219): a log-likelihood function with high curvature (large Hessian) will result in a low variance estimate, since the parameters are "well determined" by the data, and hence robust to repeated sampling.


### Boostrap Approxmation of the sampling distribution of any estimator

Idea: Treat the observed dataset as empirical approximation of true dataset. 
- Sample with replacement. Draw $N$ data points from original of size $N$ with replacement
- Repeat proces $S$ times
- For each $S$ estimate the MLE resulting in set of estimates $\hat{\theta_S}$
- The empirical distribution of these bootstrap estimates serves as an approximation of estimator's true sampling distirbution!!

#### Confidence Intervals

$$
\begin{equation*}
I=\left(\hat{\theta}-z_{\alpha / 2} \hat{\mathrm{se}}, \hat{\theta}+z_{\alpha / 2} \hat{\mathrm{se}}\right) 
\end{equation*}
$$

where $z_{\alpha / 2}$ is the $\alpha / 2$ quantile of the Gaussian cdf. If we set $\alpha=0.05$, we have $z_{\alpha / 2}=1.96$, which justifies the common approximation $\hat{\theta} \pm 2 \hat{\mathrm{se}}$.
- We can use bootstrap if not a good estimate using Gaussian
Issues
- Instead the frequentist approach just means that the procedure for generating CIs will contain the true value 95% of the time. It is commonly believed that a 95% confidence interval I for a parameter estimate θ given data D means that the true parameter lies in this interval with probability 0.95, -> This is what Bayesian Credible Interval computes


### MSE

MSE = Variance + Bias^2

$$\mathbb{E}_{\mathcal{D}, y}\left[\left(y-\hat{f}_{\mathcal{D}}(\vec{x})\right)^2\right]=(f(\vec{x})-\bar{f}(\vec{x}))^2+\mathbb{E}_{\mathcal{D}}\left[\left(\bar{f}(\vec{x})-\hat{f}_{\mathcal{D}}(\vec{x})\right)^2\right]+\mathbb{E}_\epsilon\left[\epsilon^2\right]$$
- Target - Approximation 
	- Bias: Expected Error due to model mismatch
		- True function - average of approximation function
	- Variance: Average value from estimator - approximation function
	- Noise
		- Limits error on upper bound of model
- $\begin{aligned}&\text { The bias of an estimator is defined as }\\&\operatorname{bias}(\hat{\theta}(\cdot)) \triangleq \mathbb{E}[\hat{\theta}(\mathcal{D})]-\theta^*\end{aligned}$
- Variance = $\mathbb{V}[\hat{\theta}] \triangleq \mathbb{E}\left[\hat{\theta}^2\right]-(\mathbb{E}[\hat{\theta}])^2$

Examples
- Low bias and low variance: linear regression on linear data
- low bias and high variance: high degree polynomial

A MAP estimate may be biased but have lower variance compared to an unbiased estimator. Thus we see that, provided the prior strength is properly “tuned”, a MAP estimate can outperform an ML estimate in terms of minimizing MSE.