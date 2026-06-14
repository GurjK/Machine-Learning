Factor analysis is a probabilistic model, meaning it can be treated as a building block for more complex models.
- Can be viewed as a low rank version of the gaussian distribution
##### Generative Model

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

Simplifications
- $\mu_0 = 0
