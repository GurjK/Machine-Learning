## Overview

Principal Component Analysis: Find a linear and orthogonal projection of the high dimensional data $x \in R^D$ to a lower dimensional subspace $z \in R^L$

This is a good approximation in the following sense:
- Project or encode $x$ to get $z = W^Tx$
- unproject or decode $z$ to get $\hat{x} = Wz$
- Goal is to have $\hat{x}$ to be as close to $x$ in $\mathcal{l}_2$ space

The reconstruction error is: $$
\mathcal{L}(\mathbf{W}) \triangleq \frac{1}{N} \sum_{n=1}^N\left\|\boldsymbol{x}_n-\operatorname{decode}\left(\operatorname{encode}\left(\boldsymbol{x}_n ; \mathbf{W}\right) ; \mathbf{W}\right)\right\|_2^2
$$
This function is minimized bet setting $\hat{W} = U_L$ where $U_L$ is the $L$ eigenvectors with the largest eigenvalues of the empirical covariance matrix: $$
\hat{\boldsymbol{\Sigma}}=\frac{1}{N} \sum_{n=1}^N\left(\boldsymbol{x}_n-\overline{\boldsymbol{x}}\right)\left(\boldsymbol{x}_n-\overline{\boldsymbol{x}}\right)^{\boldsymbol{\top}}=\frac{1}{N} \mathbf{X}_c^{\boldsymbol{\top}} \mathbf{X}_c
$$
- $X_c$ is a centered version of the design matrix $N \times D$

This is the equivalent to maximizing the likelihood of a latent linear Gaussian model known as probablistic PCA.

## Algorithm Derivation

Definitions:
- $\mathcal{D} = {x_n: n=1:N}$ where $x_n \in R^D$  is the data
	- $N \times D$ data matrix $X$
- $\overline{\boldsymbol{x}}=\frac{1}{N} \sum_{n=1}^N \boldsymbol{x}_n=\mathbf{0}$
	- Data matrix centered
- $z_n$: low dimensional representation of $x_n$ known as the latent factor

Goal: Approximate $x_n$ by creating a lower dimensional representation $z_n \in R^L$
- Assume that $x_n$ can be explained in terms of a weighted combinanation of basis functions: $w_1,...,w_L$
	- $\boldsymbol{x}_n \approx \sum_{k=1}^L z_{n k} \boldsymbol{w}_k$

There error can therefore be approximated by: $$
\mathcal{L}(\mathbf{W}, \mathbf{Z})=\frac{1}{N}\left\|\mathbf{X}-\mathbf{Z} \mathbf{W}^{\top}\right\|_F^2=\frac{1}{N}\left\|\mathbf{X}^{\top}-\mathbf{W} \mathbf{Z}^{\top}\right\|_F^2=\frac{1}{N} \sum_{n=1}^N\left\|\boldsymbol{x}_n-\mathbf{W} \boldsymbol{z}_n\right\|^2
$$
- $Z$ : containts low dimension versions of the rows of $X$

Goal is to minimize this error subject to constraint that $W$ is orthogonal:
- This means that, $w_1^T w_w = 1$

$$
\begin{aligned}
\mathcal{L}\left(\boldsymbol{w}_1, \tilde{\mathbf{z}}_1\right) & =\frac{1}{N} \sum_{n=1}^N\left\|\boldsymbol{x}_n-z_{n 1} \boldsymbol{w}_1\right\|^2=\frac{1}{N} \sum_{n=1}^N\left(\boldsymbol{x}_n-z_{n 1} \boldsymbol{w}_1\right)^{\top}\left(\boldsymbol{x}_n-z_{n 1} \boldsymbol{w}_1\right) \\
& =\frac{1}{N} \sum_{n=1}^N\left[\boldsymbol{x}_n^{\top} \boldsymbol{x}_n-2 z_{n 1} \boldsymbol{w}_1^{\top} \boldsymbol{x}_n+z_{n 1}^2 \boldsymbol{w}_1^{\top} \boldsymbol{w}_1\right] \\
& =\frac{1}{N} \sum_{n=1}^N\left[\boldsymbol{x}_n^{\top} \boldsymbol{x}_n-2 z_{n 1} \boldsymbol{w}_1^{\top} \boldsymbol{x}_n+z_{n 1}^2\right]
\end{aligned}
$$
- Now we can take the derivatives wrt to $z_{b1}
## Computational Issues

