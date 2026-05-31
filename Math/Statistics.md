## Covariance 
- $Cov(X,Y) = E[XY] - E[X]E[Y]$
- $Cov(Ax+b) = A Cov(x)A^T$
- 

Notes
- Uncorrelated does not imply indepdent
	- Independence implied uncorrelated
- Correlation does not imply causation
	- Spurious Correlation: Hidden common cause


## Exponential Family

Consider a family of probability distributions parameterized by $\eta \in \mathbb{R}^K$ with fixed support over $\mathcal{Y}^D \subseteq \mathbb{R}^D$. We say that the distribution $p(\boldsymbol{y} \mid \boldsymbol{\eta})$ is in the exponential family if its density can be written in the following way:

$$
\begin{equation*}
p(\boldsymbol{y} \mid \boldsymbol{\eta}) \triangleq \frac{1}{Z(\boldsymbol{\eta})} h(\boldsymbol{y}) \exp \left[\boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y})\right]=h(\boldsymbol{y}) \exp \left[\boldsymbol{\eta}^{\top} \mathcal{T}(\boldsymbol{y})-A(\boldsymbol{\eta})\right] 
\end{equation*}
$$

where $h(\boldsymbol{y})$ is a scaling constant (also known as the base measure, often 1 ), $\mathcal{T}(\boldsymbol{y}) \in \mathbb{R}^K$ are the sufficient statistics, $\boldsymbol{\eta}$ are the natural parameters or canonical parameters, $Z(\boldsymbol{\eta})$ is a normalization constant known as the partition function, and $A(\boldsymbol{\eta})=\log Z(\boldsymbol{\eta})$ is the $\log$ partition function. One can show that $A$ is a convex function over the convex set $\Omega \triangleq\left\{\boldsymbol{\eta} \in \mathbb{R}^K: A(\boldsymbol{\eta})<\infty\right\}$.

The equation can be generalized by defining $\boldsymbol{\eta}=f(\boldsymbol{\phi})$, where $\phi$ is some other, possibly smaller, set of parameters. In this case, the distribution has the form

$$
\begin{equation*}
p(\boldsymbol{y} \mid \boldsymbol{\phi})=h(\boldsymbol{y}) \exp \left[f(\boldsymbol{\phi})^{\boldsymbol{\top}} \mathcal{T}(\boldsymbol{y})-A(f(\boldsymbol{\phi}))\right] 
\end{equation*}
$$
- This can be a nonlinear transformation which would make this a curved exponential family
- Canonical Form means: $\text { If } \boldsymbol{\eta}=f(\phi)=\phi$
- Natural Exponential Family: $\mathcal{T}(y) = y$:

### Mixture Models

One way to create more complex probability models is to take a convex combination of distributions. This is called a mixture model. This has the form

$$
p(\boldsymbol{y} \mid \boldsymbol{\theta})=\sum_{k=1}^K \pi_k p_k(\boldsymbol{y})
$$
- $\pi$ =mixture weights which sum to 1
- $p$ = kth mixture component

Introduce latent variable $z \in (1,...,K)$ such that $p(z=k \mid \boldsymbol{\theta})=\pi_k$, and the conditional is $p(\boldsymbol{y} \mid z=k, \boldsymbol{\theta})=p_k(\boldsymbol{y})=p\left(\boldsymbol{y} \mid \boldsymbol{\theta}_k\right)$. That is, we define the following joint model:

$$
\begin{aligned}
p(z \mid \boldsymbol{\theta}) & =\operatorname{Cat}(z \mid \boldsymbol{\pi}) \\
p(\boldsymbol{y} \mid z=k, \boldsymbol{\theta}) & =p\left(\boldsymbol{y} \mid \boldsymbol{\theta}_k\right)
\end{aligned}
$$

where $\boldsymbol{\theta}=\left(\pi_1, \ldots, \pi_K, \boldsymbol{\theta}_1, \ldots, \boldsymbol{\theta}_K\right)$ are all the model parameters.
- Distribution is like a mixture of multiple normal distributions
- To generate a data point
	- randomly chooke a component $k$ using probabilities $\pi$
	- sample from the component distribution

## Probabiltistic graphical Models


#