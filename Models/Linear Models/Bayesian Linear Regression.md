How to compute posterior over the parameters $p(\boldsymbol{\theta} \mid \mathcal{D})$
- Assume that variance is known and just want to compute: $p\left(\boldsymbol{w} \mid \mathcal{D}, \sigma^2\right) .$

##### Priors

Assume a Gaussian Prior: $p(\boldsymbol{w})=\mathcal{N}(\boldsymbol{w} \mid \breve{\boldsymbol{w}}, \breve{\boldsymbol{\Sigma}})$

#### Posterior

Rewrite the likelihood in terms of an Multivarite Normal Distribution so 

$$
p\left(\mathcal{D} \mid \boldsymbol{w}, \sigma^2\right)=\prod_{n=1}^N p\left(y_n \mid \boldsymbol{w}^{\top} \boldsymbol{x}, \sigma^2\right)=\mathcal{N}\left(\boldsymbol{y} \mid \mathbf{X} \boldsymbol{w}, \sigma^2 \mathbf{I}_N\right)
$$

where $\mathbf{I}_N$ is the $N \times N$ identity matrix. We can then use Bayes rule for Gaussians to derive the posterior, which is as follows:

$$
\begin{aligned}
p\left(\boldsymbol{w} \mid \mathbf{X}, \boldsymbol{y}, \sigma^2\right) & \propto \mathcal{N}(\boldsymbol{w} \mid \breve{\boldsymbol{w}}, \breve{\boldsymbol{\Sigma}}) \mathcal{N}\left(\boldsymbol{y} \mid \mathbf{X} \boldsymbol{w}, \sigma^2 \mathbf{I}_N\right)=\mathcal{N}(\boldsymbol{w} \mid \overparen{\boldsymbol{w}}, \overparen{\boldsymbol{\Sigma}}) \\
\overparen{\boldsymbol{w}} & \triangleq \mathbb{\Sigma}\left(\breve{\boldsymbol{\Sigma}}^{-1} \breve{\boldsymbol{w}}+\frac{1}{\sigma^2} \mathbf{X}^{\top} \boldsymbol{y}\right) \\
\overparen{\boldsymbol{\Sigma}} & \triangleq\left(\breve{\boldsymbol{\Sigma}}^{-1}+\frac{1}{\sigma^2} \mathbf{X}^{\top} \mathbf{X}\right)^{-1}
\end{aligned}
$$
- where $\widehat{\boldsymbol{w}}$ is the posterior mean, and $\widehat{\boldsymbol{\Sigma}}$ is the posterior covariance.

#### Computing the Posterior Predictive

Above is discussed about parameters in the model but need to determine uncertainity in prediction:

Posterior Predictive Distribution would be : $$
\begin{aligned}
p\left(y \mid \boldsymbol{x}, \mathcal{D}, \sigma^2\right) & =\int \mathcal{N}\left(y \mid \boldsymbol{x}^{\boldsymbol{\top}} \boldsymbol{w}, \sigma^2\right) \mathcal{N}(\boldsymbol{w} \mid \widehat{\boldsymbol{\mu}}, \widehat{\mathbf{\Sigma}}) d \boldsymbol{w} \\
& =\mathcal{N}\left(y \mid \widehat{\boldsymbol{\mu}}^{\boldsymbol{\top}} \boldsymbol{x}, \widehat{\sigma}^2(\boldsymbol{x})\right)
\end{aligned}
$$
where $\widehat{\sigma}^2(\boldsymbol{x}) \triangleq \sigma^2+\boldsymbol{x}^{\top} \boldsymbol{\Sigma} \boldsymbol{x}$ is the variance of the posterior predictive distribution at point $\boldsymbol{x}$ after seeing the $N$ training examples. The predicted variance depends on two terms: the variance of the observation noise, $\sigma^2$, and the variance in the parameters, $\widehat{\Sigma}$. The latter translates into variance about observations in a way which depends on how close $\boldsymbol{x}$ is to the training data $\mathcal{D}$.  This can be important for certain applications, such as active learning, where we choose where to collect training data.

Some cases can't compute $p(w|D)$ therefore we compute a point estimate as follows: $$p\left(y \mid \boldsymbol{x}, \mathcal{D}, \sigma^2\right)=\int \mathcal{N}\left(y \mid \boldsymbol{x}^{\top} \boldsymbol{w}, \sigma^2\right) \delta(\boldsymbol{w}-\hat{\boldsymbol{w}}) d \boldsymbol{w}=p\left(y \mid \boldsymbol{x}^{\top} \hat{\boldsymbol{w}}, \sigma^2\right)$$
- This is a constant and independent of the data

Why Center the Data?
- Correlation between parameters causes computational difficulties
	- To understand why this happens, note that each data point induces a likelihood function corresponding to a line which goes through that data point. When we look at all the data together, we see that predictions with maximum likelihood must correspond to lines that go through the mean of the data, $(\bar{x}, \bar{y})$. There are many such lines, but if we increase the slope, we must decrease the intercept. Thus we can think of the set of high probability lines as spinning around the data mean, like a wheel of fortune. ${ }^5$ This correlation between $w_0$ and $w_1$ is why the posterior has the form of a diagonal line. (The Gaussian prior converts this into an elongated ellipse, but the posterior correlation still persists until the sample size causes the posterior to shrink to a point.)
- Centering input allows lines to pivot arround origin reducing posterior correlation between $w_0$ and $w_1$

How to deal with MultiCollinearity?
- Including all input variables does not harm predicability only interpretabilitiy
- We might see that value of knowing one variable means that other variable is worthless based on coefficients but if we simply rerun with different data we would get opposite conclusiomn



##### ARD
Automatic Relevancy Determination or Sparse Bayesian Learning

Consider a linear regression model with known observation noise but unknown regression weights, $\mathcal{N}\left(\boldsymbol{y} \mid \mathbf{X} \boldsymbol{w}, \sigma^2 \mathbf{I}\right)$. Suppose we use a Gaussian prior for the weights, $w_j \sim \mathcal{N}\left(0,1 / \alpha_j\right)$, where $\alpha_j$ is the precision of the jthj parameter
- $\hat{\boldsymbol{\alpha}}=\underset{\boldsymbol{\alpha}}{\operatorname{argmax}} p(\boldsymbol{y} \mid \mathbf{X}, \boldsymbol{\alpha})$
- Marginal Likelihood: $p(\boldsymbol{y} \mid \mathbf{X}, \boldsymbol{\alpha})=\int p\left(\boldsymbol{y} \mid \mathbf{X} \boldsymbol{w}, \sigma^2\right) p\left(\boldsymbol{w} \mid \mathbf{0}, \operatorname{diag}(\boldsymbol{\alpha})^{-1}\right) d \boldsymbol{w}$
- MAP estimate: $\hat{\boldsymbol{w}}=\underset{\boldsymbol{w}}{\operatorname{argmax}} \mathcal{N}\left(\boldsymbol{w} \mid \mathbf{0}, \hat{\boldsymbol{\alpha}}^{-1}\right)$

Large $\alpha_j$
- Tiny variance
- pushes $w_j \to 0$

Small $\alpha_j$
- Large variance
- Coefficient Freedom
Model automatically decides relevant and irrelevant features