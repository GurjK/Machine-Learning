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
