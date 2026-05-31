
Alternative to Group
- We don't know the groups ahead of time but we desire that highly correlated coefficients be treated as an implicity group
- Solution: Elastic Net: Hybrid of Lasso and Ridge Regression

Loss Function: $\mathcal{L}\left(\boldsymbol{w}, \lambda_1, \lambda_2\right)=\|\boldsymbol{y}-\mathbf{X} \boldsymbol{w}\|^2+\lambda_2\|\boldsymbol{w}\|_2^2+\lambda_1\|\boldsymbol{w}\|_1$
- Strictly Convex so Minimum exists even if $X$ is not full rank
	- Any strictly convex penalty on $w$ will exhibity grouping effect
	- regressions coefficients of highly correlated variables tend to be equal
		- Lasso would eliminate one or the other
- $D>N$ then maximum number of nonzero elements selected is $N$

#### Optimization Algos for Lasso, L1 objectives

1. Coordinate Descent
2. Proximal Gradient Descent
3. LARS
4. 