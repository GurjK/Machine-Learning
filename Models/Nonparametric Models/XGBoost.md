Extreme Gradient Boosting
- add regularizer on the tree complexity
- Use second order approximation of the loss
- Samples features at internal nodes
- Various computer science methods (handling out of core computation for large datasets)

Loss Function: $$
\mathcal{L}(f)=\sum_{i=1}^N \ell\left(y_i, f\left(\boldsymbol{x}_i\right)\right)+\Omega(f)
$$
- Objective has two two parts, training loss and regularization
	- Training Loss: How well model fits training data
		- Squared Loss function!
	- Regularization: Complexity of the trees: $\Omega(f)$
- $\Omega(f)=\gamma J+\frac{1}{2} \lambda \sum_{j=1}^J w_j^2$
	- regularize on Tree complexity
		- Standard trees can memorize the noise in the tree leading to overfitting. 
		- Want simplest tree that does a good job
	- $J$ is the number of leaves
	- $\gamma \geq 0$ regularization coefficients
		- gamma: Leaf penalty
		- minimum amount of loss reduction on a new split that must be met to justify the added complexity of one more leaf
	- $\lambda \geq 0$ regularization coefficients
		- lambda
		- L2 regularization on leaf scores($w_j$) weights
		- Shrinks the final prediction value of each leaf making model more conservative
	- $w_j$
		- Leaf Scores or weights
- $\sum_{i=1}^N \ell\left(y_i, f\left(\boldsymbol{x}_i\right)\right)$ : Training Loss
	- Additive Boosting
Example:
- Split 1 age < 15
	- Left: is male
		- Left : Leaf 1: w1 = 2
		- Right: Leaf 2: w2 = .1
	- Right: Leaf 3: w3 = -1
- $\Omega(f) = \gamma * 3 + .5 \lambda (2^2+.1^2+ (-1)^2)$


The goal is to find the optimal split and leaf weight. To do this we need the direction and curvature, so gradient and Hessian. 

At the $m$th step the loss is: $$
\mathcal{L}_m\left(F_m\right)=\sum_{i=1}^N \ell\left(y_i, f_{m-1}\left(\boldsymbol{x}_i\right)+F_m\left(\boldsymbol{x}_i\right)\right)+\Omega\left(F_m\right)+\text { const }
$$
Now we can compute a seond order taylor approximation:
$$
\mathcal{L}_m\left(F_m\right) \approx \sum_{i=1}^N\left[\ell\left(y_i, f_{m-1}\left(\boldsymbol{x}_i\right)\right)+g_{i m} F_m\left(\boldsymbol{x}_i\right)+\frac{1}{2} h_{i m} F_m^2\left(\boldsymbol{x}_i\right)\right]+\Omega\left(F_m\right)+\text { const }
$$
- $h_{im}$ : Hessian: $h_{i m}=\left[\frac{\partial^2 \ell\left(y_i, f\left(\boldsymbol{x}_i\right)\right)}{\partial f\left(\boldsymbol{x}_i\right)^2}\right]_{f=f_{m-1}}$
- $F(x) = w_{q(x)}$ where
- $q$ specifies which leaf node $x$ belongs to 
- $w$ are leaf weights

Rewrite the Loss as $$
\begin{aligned}
\mathcal{L}_m(q, \boldsymbol{w}) & \approx \sum_{i=1}^N\left[g_{i m} F_m\left(\boldsymbol{x}_i\right)+\frac{1}{2} h_{i m} F_m^2\left(\boldsymbol{x}_i\right)\right]+\gamma J+\frac{1}{2} \lambda \sum_{j=1}^J w_j^2 \\
& =\sum_{j=1}^J\left[\left(\sum_{i \in I_j} g_{i m}\right) w_j+\frac{1}{2}\left(\sum_{i \in I_j} h_i+\lambda\right) w_j^2\right]+\gamma J
\end{aligned}
$$
- $I_j=\left\{i: q\left(\boldsymbol{x}_i\right)=j\right\}$ is the set of indices of data points assigned to the $j$ 'th leaf. us define $G_{j m}=\sum_{i \in I_j} g_{i m}$ and $H_{j m}=\sum_{i \in I_j} h_{i m}$. Then the above simplifies to

$$
\mathcal{L}_m(q, \boldsymbol{w})=\sum_{j=1}^J\left[G_{j m} w_j+\frac{1}{2}\left(H_{j m}+\lambda\right) w_j^2\right]+\gamma J
$$
- Optimal weights: $w_j^*=-\frac{G_{j m}}{H_{j m}+\lambda}$
- Loss at different tree structures $q$ then becomes: $$
\mathcal{L}_m\left(q, \boldsymbol{w}^*\right)=-\frac{1}{2} \sum_{j=1}^J \frac{G_{j m}^2}{H_{j m}+\lambda}+\gamma J
$$
If we split a leaf into left and right then we can compute the gain as 
$$
\text { gain }=\frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{\left(G_L+G_R\right)^2}{\left(H_L+H_R\right)+\lambda}\right]-\gamma
$$
- where $G_L=\sum_{i \in I_L} g_{i m}, G_R=\sum_{i \in I_R} g_{i m}, H_L=\sum_{i \in I_L} h_{i m}$, and $H_R=\sum_{i \in I_R} h_{i m}$.
- Not worth splitting the node if the gain is negative