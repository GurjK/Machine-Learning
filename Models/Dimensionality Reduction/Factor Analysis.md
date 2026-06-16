Factor analysis is a probabilistic model, meaning it can be treated as a building block for more complex models.
- Can be viewed as a low rank version of the gaussian distribution
##### Generative Model

2 Main Steps
- Generate your latent factors, $z$
- Generate Observations: $x$

Factor Analysis corresponds to the following linear-Gaussian latent variable generative model:
- $p(\boldsymbol{z})=\mathcal{N}\left(\boldsymbol{z} \mid \boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0\right)$
	- Prior on latent factor
	- Simplification: $\mathrm{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
		- Latent factor have no mean, unit variance and are independent a priori
- $p(\boldsymbol{x} \mid \boldsymbol{z}, \boldsymbol{\theta})=\mathcal{N}(\boldsymbol{x} \mid \mathbf{W} \boldsymbol{z}+\boldsymbol{\mu}, \boldsymbol{\Psi})$
	- Oberservation Model: $x = \mu + Wz + \epsilon$
	- $\epsilon \sim N(0,\Psi)$
	- $W = D \times L$ matrix know as the factor loading matrix
	- $\Psi$ is a diagonal $D \times D$ covariance matrix
	- Each variable gets its own independent noise
	- $x \in R^D$ : high dimensional observed vector
	- $z \in R^L$: hidden factor trying to reduce $x$

Math Notes
- $\mathbb{E}[\boldsymbol{x}]=\mathbf{W} \boldsymbol{\mu}_0+\boldsymbol{\mu}$ 
- $\operatorname{Cov}[\boldsymbol{x}]=\mathbf{W} \operatorname{Cov}[\boldsymbol{z}] \mathbf{W}^{\top}+\boldsymbol{\Psi}=\mathbf{W} \boldsymbol{\Sigma}_0 \mathbf{W}^{\top}+\boldsymbol{\Psi}$
- $\mathbb{V}\left[x_d\right]=\sum_{k=1}^L w_{d k}^2+\psi_d$,
	- First term is variance due to common factors and secon is called uniqueness

Simplifications
- $\mu_0 = 0$
- $\Sigma_0 = I$
- $\tilde{\mathbf{W}}=\mathbf{W} \boldsymbol{\Sigma}_0^{-\frac{1}{2}}$

After simplifying we have: $$
\begin{aligned}
p(\boldsymbol{z}) & =\mathcal{N}(\boldsymbol{z} \mid \mathbf{0}, \mathbf{I}) \\
p(\boldsymbol{x} \mid \boldsymbol{z}) & =\mathcal{N}(\boldsymbol{x} \mid \mathbf{W} \boldsymbol{z}+\boldsymbol{\mu}, \boldsymbol{\Psi}) \\
p(\boldsymbol{x}) & =\mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}, \mathbf{W} \mathbf{W}^{\top}+\boldsymbol{\Psi}\right)
\end{aligned}
$$
Generally, FA approximates Covariance matrix using a low rank decomposition: $\mathbf{C}=\operatorname{Cov}[\boldsymbol{x}]=\mathbf{W W}^{\top}+\boldsymbol{\Psi}$
- $\Psi$ must be diagonal
	- After accounting for latent factors, any remaining noise is independent. All correlations between variables must be explained through the latent factors
- Covariance = Low Rank part + Diagonal part

Using Bayes Rule we get: $$
p(\boldsymbol{z} \mid \boldsymbol{x})=\mathcal{N}\left(\boldsymbol{z} \mid \mathbf{W}^{\top} \mathbf{C}^{-1}(\boldsymbol{x}-\boldsymbol{\mu}), \mathbf{I}-\mathbf{W}^{\top} \mathbf{C}^{-1} \mathbf{W}\right)
$$
- $E[z \mid x]=W^T C^{-1}(x-\mu)$
	- Factor Score. Best Estimate of the hidden factors given the observed data
	- Different between PCA and FA?
		- FA accounts for noise covariance $\Psi$
### Probabilistic PCA

Assume:
- $W$ has orthonormal columns
- $\Psi = \sigma^2 I$
- 

This is called Probablistic principal component analysis or sensible PCA
- PCA assumes no probablistic covariance model
	- Can't handle missing values
	- PCA is a limiting case of PPCA wehere $\sigma^2 \to 0$
- PPCA: $WW^T + \sigma^2 I$
	- Can handle them naturally using EM because it has a likelihood function.
- FA: $WW^T + \Psi$

Marginal Distribution: $p(\boldsymbol{x} \mid \boldsymbol{\theta})=\int \mathcal{N}\left(\boldsymbol{x} \mid \mathbf{W} \boldsymbol{z}, \sigma^2 \mathbf{I}\right) \mathcal{N}(\boldsymbol{z} \mid \mathbf{0}, \mathbf{I}) d \boldsymbol{z}=\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}, \mathbf{C})$
- $\mathbf{C}=\mathbf{W} \mathbf{W}^{\top}+\sigma^2 \mathbf{I}$
Log Likelihood: $\log p\left(\mathbf{X} \mid \boldsymbol{\mu}, \mathbf{W}, \sigma^2\right)=-\frac{N}{2}\left[D \log (2 \pi)+\log |\mathbf{C}|+\operatorname{tr}\left(\mathbf{C}^{-1} \mathbf{S}\right)\right]$
- $S$: Empirical Covariance matrix

$W$ must satisfy
- $\mathbf{W}=\mathbf{U}_L\left(\mathbf{L}_L-\sigma^2 \mathbf{I}\right)^{\frac{1}{2}} \mathbf{R}$
	- - $U_L=$ leading PCA eigenvectors
	- $\Lambda_L=$ leading eigenvalues
	- $R=$ arbitrary orthogonal matrix
- In PCA, $W = U_L L_L^{1/2}$
	- which is the average distortion associated with the discarded dimensions.


To use PPCA as an alternative to PCA, we need to compute the posterior mean: $E[Z |x]$, which is the equivalent of the encoder model.
- $\mathbb{E}[\boldsymbol{z} \mid \boldsymbol{x}]=\left(\mathbf{W}^{\boldsymbol{\top}} \mathbf{W}\right)^{-1} \mathbf{W}^{\boldsymbol{\top}}(\boldsymbol{x}-\overline{\boldsymbol{x}})$ which is the orthogonal projection of the data into the latent space, as in standard PCA.

PCA: $z_i=W^T x_i$
- Point Estimate
PPCA: $p\left(z_i \mid x_i\right)=N\left(m_i, S_i\right)$
- estimated factor values and uncertainty around that value
- Probability distribution so u can sample new observations

```
#PCA - sklearn
#PPCA - ppca-rs
```

**How to Generate a Unique W**
- Force W to have orthonormla columns
	- $W^TW=I$
	- 
- Force W to be lower trianglular
- Use L1 regularization to introduce sparse factor analysis
- An informative rotation matrix
- Use of non Gaussian priors for the latent factors

### Non-Linear Factor Analysis

Assumes the observed data can be modelled arising from a linear mapping from a low-dimensional set of Gaussian factors. Relax this by letting mapping from $z$ to $x$ be a non-linear model such as a neural network. Then the model becomes: $$
p(\boldsymbol{x})=\int \mathcal{N}(\boldsymbol{x} \mid f(\boldsymbol{z} ; \boldsymbol{\theta}), \boldsymbol{\Psi}) \mathcal{N}(\boldsymbol{z} \mid \mathbf{0}, \mathbf{I}) d \boldsymbol{z}
$$
Unfortunately we can no longer compute the posterior or the MLE exactly, so we need to use approximate methods. Variational autoencoders, which is the most common way to approximate a nonlinear FA model.

### Mixture of factor analyzers

Limitation: Factor analysis arises from a linear mapping from a low dimensional set of Gaussian factors
- Solution: Assume only locally linear so overall model is a weighted combination of mixture factor, **mixture of factor analyzers**

Latent indictor: $m_n$ specifies subspace to generate data point $n$
- $m_n =k$, sample $z_n$ from a gaussian prior and pass it through to $W_k$ matrix and add noise, 
	- $k$ is number of mixture components
- $W_k$ mapping from L to D dimensional visible space
$$
\begin{aligned}
p\left(\boldsymbol{x}_n \mid \boldsymbol{z}_n, m_n=k, \boldsymbol{\theta}\right) & =\mathcal{N}\left(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k+\mathbf{W}_k \boldsymbol{z}_n, \boldsymbol{\Psi}_k\right) \\
p\left(\boldsymbol{z}_n \mid \boldsymbol{\theta}\right) & =\mathcal{N}\left(\boldsymbol{z}_n \mid \mathbf{0}, \mathbf{I}\right) \\
p\left(m_n \mid \boldsymbol{\theta}\right) & =\operatorname{Cat}\left(m_n \mid \boldsymbol{\pi}\right)
\end{aligned}
$$
Distribution:$$
p(\boldsymbol{x} \mid \boldsymbol{\theta})=\sum_k p(c=k) \int d \boldsymbol{z} p(\boldsymbol{z} \mid c) p(\boldsymbol{x} \mid \boldsymbol{z}, c)=\sum_k \pi_k \int d \boldsymbol{z} \mathcal{N}\left(\boldsymbol{z} \mid \boldsymbol{\mu}_k, \mathbf{I}\right) \mathcal{N}\left(\boldsymbol{x} \mid \mathbf{W} \boldsymbol{z}, \sigma^2 \mathbf{I}\right)
$$
- If $\Psi = \sigma^2 I$, get a mixture of PPCA models 

### Exponential Family Factor Analysis

Replace Gaussian output, with a suitable member of the exponential family
- natural parameters are given by a linear function of $z_n$
- $p\left(\boldsymbol{x}_n \mid \boldsymbol{z}_n\right)=\exp \left(\mathcal{T}(\boldsymbol{x})^{\boldsymbol{\top}} \boldsymbol{\theta}+h(\boldsymbol{x})-g(\boldsymbol{\theta})\right)$
	- where the $N \times D$ matrix of natural parameters is assumed to be given by the low rank decomposition $\mathbf{\Theta}=\mathbf{Z} \mathbf{W}$, where $\mathbf{Z}$ is $N \times L$ and $\mathbf{W}$ is $L \times D$. The resulting model is called exponential family factor analysis.

Issues
- cannot compute exact posterior

### Other forms of Factor Analysis
- Use exponential family PCA
- categorical PCA
- binary PCA

### Factor Analysis for Paired Data
- supervised PCA
	- model the joint $p(x,y)$
	- For paired data
	- Leads to Bayesian Factor Regression
- Partial Least Squares
	- To improve predictive nature
	- Add an extra latent variable $z_n^x$ just for inputs that is different for $z_n^s$ that is the shared bottleneck for $x_n$ and $y_n$
- Canonical Correlation Analysis
	- Fully Symmetric model, so we can capture dependence between $x$ and $y$ while allowing for domain specific or private noise sources
	- $z_n^x$ for x and $z_n^y$ for y, shared $z_n^s$