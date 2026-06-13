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
- Now we can take the derivatives wrt to $z_{n1}$ to get and using the orthonormla arguement that : 

$$
\frac{\partial}{\partial z_{n 1}} \mathcal{L}\left(\boldsymbol{w}_1, \tilde{\mathbf{z}}_1\right)=\frac{1}{N}\left[-2 \boldsymbol{w}_1^{\top} \boldsymbol{x}_n+2 z_{n 1}\right]=0 \Rightarrow z_{n 1}=\boldsymbol{w}_1^{\top} \boldsymbol{x}_n
$$
Conclusion:
- Optimal embedding is determined by orthogonally projecting the data onto $w_1$.
Now we can plug this into the loss to get the following:

$$
\begin{aligned}
&\mathcal{L}\left(\boldsymbol{w}_1\right)=\mathcal{L}\left(\boldsymbol{w}_1, \tilde{\mathbf{z}}_1^*\left(\boldsymbol{w}_1\right)\right)=\frac{1}{N} \sum_{n=1}^N\left[\boldsymbol{x}_n^{\top} \boldsymbol{x}_n-z_{n 1}^2\right]=\text { const }-\frac{1}{N} \sum_{n=1}^N z_{n 1}^2\\
&\text { To solve for } \boldsymbol{w}_1 \text {, note that }\\
&\mathcal{L}\left(\boldsymbol{w}_1\right)=-\frac{1}{N} \sum_{n=1}^N z_{n 1}^2=-\frac{1}{N} \sum_{n=1}^N \boldsymbol{w}_1^{\top} \boldsymbol{x}_n \boldsymbol{x}_n^{\top} \boldsymbol{w}_1=-\boldsymbol{w}_1^{\top} \hat{\boldsymbol{\Sigma}} \boldsymbol{w}_1
\end{aligned}
$$
- $\Sigma$ is the covariance matrix empirically
	- Data is centered
- impose the contraint the $||w_1|| =1$ and instead optimize 
$$
\tilde{\mathcal{L}}\left(\boldsymbol{w}_1\right)=\boldsymbol{w}_1^{\top} \hat{\boldsymbol{\Sigma}} \boldsymbol{w}_1-\lambda_1\left(\boldsymbol{w}_1^{\top} \boldsymbol{w}_1-1\right)
$$
- $\lambda$ is a lagrange multiplier here!

Taking derivatives and solving gives: 
$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{w}_1} \tilde{\mathcal{L}}\left(\boldsymbol{w}_1\right) & =2 \hat{\boldsymbol{\Sigma}} \boldsymbol{w}_1-2 \lambda_1 \boldsymbol{w}_1=0 \\
\hat{\boldsymbol{\Sigma}} \boldsymbol{w}_1 & =\lambda_1 \boldsymbol{w}_1
\end{aligned}
$$

*Hence the optimal direction onto which we should project the data is an eigenvector of the covariance matrix.*
- We want to maximize $\boldsymbol{w}_1^{\top} \hat{\boldsymbol{\Sigma}} \boldsymbol{w}_1=\lambda_1$ which is the same as, picking the eigenvector which corresponds to the largest eigenvalue

Important Notes
- Minimizing the reconstruction error is the same thing as maximizing the variance of the projected data: $\arg \min _{\boldsymbol{w}_1} \mathcal{L}\left(\boldsymbol{w}_1\right)=\arg \max _{\boldsymbol{w}_1} \mathbb{V}\left[\tilde{\mathbf{z}}_1\left(\boldsymbol{w}_1\right)\right]$
## Computational Issues

1) Working with the correlation matrix is better then working with the covariance matirx
	1) PCA can be misled by directions in which the variance is high merely because of the scale of the data
	2) Equivalent to standardizing the data
2) If $D>N$ then it is faster to working with the $N \times N$ Gram Matrix: $XX^T$
	1) $U$ contains the eigenvectors of $XX^T$ with eigenvalues $\Lambda$
	2) $\mathbf{V}=\mathbf{X}^{\top} \mathbf{U} \boldsymbol{\Lambda}^{-\frac{1}{2}}$
		1) This is the normalized eigenvectors
		2) $\mathbf{V}=\mathbf{X}^{\top} \mathbf{U}$
3) Can be computed using the truncated SVD
	1) Can perform PCA by using eigendomposition of $\Sigma$ or SVD decomposition of $X$
		1) $\mathbf{Z}=\mathbf{X} \mathbf{W}=\mathbf{U}_X \mathbf{S}_X \mathbf{V}_X^{\top} \mathbf{V}_X=\mathbf{U}_X \mathbf{S}_X$
			1) $_X$ means the L truncated SVD approximation of data matrix $X$
		2) To get the data back: $\hat{\mathbf{X}}=\mathbf{Z} \mathbf{W}^{\top}=\mathbf{U}_X \mathbf{S}_X \mathbf{V}_X^{\top}$
	2) For very high dimensional data we can use randomized SVD algo:

### Number of Dimensions

Using L dimensions the reconstruction error is: $$
\mathcal{L}_L=\frac{1}{|\mathcal{D}|} \sum_{n \in \mathcal{D}}\left\|\boldsymbol{x}_n-\hat{\boldsymbol{x}}_n\right\|^2
$$
- $\mu$ is the empirical mean
- $\hat{\boldsymbol{x}}_n=\mathbf{W} \boldsymbol{z}_n+\boldsymbol{\mu}$
- $\boldsymbol{z}_n=\mathbf{W}^{\boldsymbol{\top}}\left(\boldsymbol{x}_n-\boldsymbol{\mu}\right)$

Methods
- Reconstruction Error plot: We can plot $\mathcal{L}_L$ vs $L$
	- Ideally plot this on the test data to avoid overfitting
- Scree Plots: plot $\lambda_j$ vs $j$
	- $\mathcal{L}_L=\sum_{j=L+1}^{D} \lambda_j$
		- As the number of dimensions increase, eigenvalues get smaller and so does the reconstruction error
	- Fraction of Varianc eExplain: $F_L=\frac{\sum_{j=1}^L \lambda_j}{\sum_{j^{\prime}=1}^{L^{\max }} \lambda_{j^{\prime}}}$
- Profile Likelihood
	- Knee or elbow point where we go from large errors to small in reconstruction plot
	- the rate of decrease in the error function will be high, whereas for L > L∗, the gains will be smaller, since the model is already sufficiently complex to capture the true distribution.
	- Profile Liklihood
		- automates the detection in the gradient of the curve
		- Which $L$ makes the observed eignevalue spectrum most likely
		- $\lambda_L$
			- Now consider partitioning these values into two groups, depending on whether $k <L$ or $k > L$, where L is some threshold which we will determine.
			- 
