Factor analysis is a probabilistic model, meaning it can be treated as a building block for more complex models.
- Can be viewed as a low rank version of the gaussian distribution
##### Generative Model

Factor Analysis corresponds to the following linear-Gaussian latent variable generative model:
- $p(\boldsymbol{z})=\mathcal{N}\left(\boldsymbol{z} \mid \boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0\right)$
	- Prior on latent factor
	- $z 
- $p(\boldsymbol{x} \mid \boldsymbol{z}, \boldsymbol{\theta})=\mathcal{N}(\boldsymbol{x} \mid \mathbf{W} \boldsymbol{z}+\boldsymbol{\mu}, \boldsymbol{\Psi})$
	- $W = D \times L$ matrix know as the factor loading matrix
	- $\Psi$ is a diagonal $D \times D$ covariance matrix
	- $x \in R^D$ : high dimensional observed vector
	- $z \in R^L$: hidden factor trying to reduce $x$

