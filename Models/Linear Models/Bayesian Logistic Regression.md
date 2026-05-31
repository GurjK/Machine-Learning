
Desire to compute the posterior in order to capture uncertaintity: $p(\boldsymbol{w} \mid \mathcal{D})$
- Wrong decision is costly 
- Little Data
- Issue: Canot compute posterior exactly for Bayesian Logistic Regression

Apprximations
- Laplace Approximations

#### Laplace Approximations

Laplace Approximation approximates posterior using a Gaussian
- Mean of Gaussian is equal to the MAP estimate $\hat{w}$
- Covariance = inverse Hessian at map estiamte: $p(\boldsymbol{w} \mid \mathcal{D}) \approx \mathcal{N}\left(\boldsymbol{w} \mid \hat{\boldsymbol{w}}, \mathbf{H}^{-1}\right)$


Approximating Posterior Predictive
- Realistically in ML we want to predict $y$ given $x$ so we want: the posterior predictive distribution: $p(y \mid \boldsymbol{x}, \mathcal{D})=\int p(y \mid \boldsymbol{x}, \boldsymbol{w}) p(\boldsymbol{w} \mid \mathcal{D}) d \boldsymbol{w}$
	- First compute $\hat{w}$ such as MLE or MAP
	- Assume $p(w | D) = \delta (w - \hat{w})$
	- Plug In Approximation: $p(\boldsymbol{y} \mid \boldsymbol{x}, \mathcal{D}) \approx \int p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{w}) \delta(\boldsymbol{w}-\hat{\boldsymbol{w}}) d \boldsymbol{w}=p(\boldsymbol{y} \mid \boldsymbol{x}, \hat{\boldsymbol{w}})$

But if we want uncertainity in our predictions should use a non-degenerate posterior:
- Common is Gaussian
- How to approximate the integral?

Approximating Integral
- Mote Carlo Approximation: $p(y=1 \mid \boldsymbol{x}, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S \sigma\left(\boldsymbol{w}_s^{\top} \boldsymbol{x}\right)$
	- Draw $S$ samples from posterior: $w_s = p(w |D)$
	- Can be slow
- Probit Approximation
	- $\begin{aligned} p(y=1 \mid \boldsymbol{x}, \mathcal{D}) & \approx \sigma(\kappa(v) m) \\ m & =\mathbb{E}[a]=\boldsymbol{x}^{\top} \boldsymbol{\mu} \\ v & =\mathbb{V}[a]=\mathbb{V}\left[\boldsymbol{x}^{\top} \boldsymbol{w}\right]=\boldsymbol{x}^{\top} \boldsymbol{\Sigma} \boldsymbol{x}\end{aligned}$
	- Predictions are less extreme in terms of confidence then plug in estimate
	- Being Bayesian does not change misclasffication rate
		- but it does change confidence estimates of the model
	- Generalized Probit Approximation: $\begin{aligned} p(y=c \mid \boldsymbol{x}, \mathcal{D}) & \approx \frac{\exp \left(\kappa\left(v_c\right) m_c\right)}{\sum_{c^{\prime}} \exp \left(\kappa\left(v_{c^{\prime}}\right) m_{c^{\prime}}\right)} \\ m_c & =\overline{\boldsymbol{m}}_c^{\top} \boldsymbol{x} \\ v_c & =\boldsymbol{x}^{\top} \mathbf{V}_{c, c} \boldsymbol{x}\end{aligned}$
		- Takses into account posterior covariance