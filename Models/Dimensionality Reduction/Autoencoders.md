PCA and Factor Analysis are a linear mapping from $x \to z$ called an encoder,$f_e$, then from $z \to x$ is called the decoder, $f_d$

Overall reconstruction function: $r(x) = f_d (f_e) (x))$
- trained to minimize $\mathcal{L}(\boldsymbol{\theta})=\|r(\boldsymbol{x})-\boldsymbol{x}\|_2^2$. More generally, we can use $\mathcal{L}(\boldsymbol{\theta})=-\log p(\boldsymbol{x} \mid r(\boldsymbol{x}))$

Consider where the enconder and decoder are nonlinear mapping implmented by neural networks
- MLP with one hidden layer, the hidden units are a low dimensional **bottleneck** between input and reconstruction
- Narrow the bottleneck layer $L <<D$
	- undercomplete
- $L >>D$
	- overcomplete

**Bottleneck Encoders**

Consider a Linear autoeoncoder where
- $z = W_1 x$: Hidden layer output
- $\hat{x} = W_2 z = W_2 W_1 x = Wx$
	- $W_1 = L \times D$
	- $W_2 = D \times L$
	- $L < D$
- Minimize: $\mathcal{L}(\mathbf{W})=\sum_{n=1}^N\left\|\boldsymbol{x}_n-\mathbf{W} \boldsymbol{x}_n\right\|_2^2$,
	- $\hat{W}$ is orthogonal projection onto the first $L$ eigenvectors of the covariance matrix of the Data
	- This is PCA!!!!

**

