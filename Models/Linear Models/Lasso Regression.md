
Goal: Parameters should not be small like in Ridge but actually zero out, therefore we want to minimize the L0-norm: $\|\boldsymbol{w}\|_0=\sum_{d=1}^D \mathbb{I}\left(\left|w_d\right|>0\right)$
- Performs Feature Selection
- L0 norm, counts the number of nonzero entries
	- Ex: $w=(3,0,0,-2.,), ||w||_0 = 3$
- This is the limit of Lasso which really uses $L_1$ regularization

#### Map Estimation with a Laplace Prior (L1 Regularization)

$$p(\boldsymbol{w} \mid \lambda)=\prod_{d=1}^D \operatorname{Laplace}\left(w_d \mid 0,1 / \lambda\right) \propto \prod_{d=1}^D e^{-\lambda\left|w_d\right|}$$
- where $\lambda$ is sparsity parameter 
- $\operatorname{Laplace}(w \mid \mu, b) \triangleq \frac{1}{2 b} \exp \left(-\frac{|w-\mu|}{b}\right)$
	- $\mu$ : location parameter
	- $b$ : scale parameter
	- The Laplace Distribution is in the form of two exponentialy distributions joined back to back. 
	- Naturally is the difference between two iid exponential random variables
		- Difference in waiting times for examples
- each coefficient $w_d$ is iid centered at 0 with a Laplace distribution
- If you put a Laplace prior on the coefficients $w_d$​, the MAP estimate becomes Lasso regression.

Objective Function becomes: $$\operatorname{PNLL}(\boldsymbol{w})=-\log p(\mathcal{D} \mid \boldsymbol{w})-\log p(\boldsymbol{w} \mid \lambda)=\|\mathbf{X} \boldsymbol{w}-\boldsymbol{y}\|_2^2+\lambda\|\boldsymbol{w}\|_1$$
- $\text { where }\|\boldsymbol{w}\|_1 \triangleq \sum^D\left|w_d\right| \text { is the } \ell_1 \text { norm of } \boldsymbol{w} \text {. }$
- Lasso: Least Abosolute shrinkage and selection operator

MAP Estimation with a Laplace Prior is called $\mathcal{l_1}$ regularlization
- Generall any norm works. For $q<1$ we get space solution where $q=0$ gives the $l_0$ norm but for $q<1$ it becomes non-convex so $l_1$ is tightest convex relation of the $l_0$ norm

#### Why Spare Solutions from $l_1$ ?

Lagrangian has the form: $\min _{\boldsymbol{w}} \operatorname{NLL}(\boldsymbol{w}) \text { s.t. }\|\boldsymbol{w}\|_1 \leq B$
- The contour plot for this is a diamond
- From contrained optimization, optimal solution occurs where lowest level set of objective function intersects the constrain surface
	- As you relax constraint $B$ is grows like a diamond until it meets objective
	- Corners more likely to interest the ellipse than one of the sides
		- corners are sparse solutions
	- L2 has no corners, so there are no sparse solutions

#### Hard and Soft Thresholding

Gradient for the smooth part of the NLL: $\mathcal{L}(\boldsymbol{w})=\operatorname{NLL}(\boldsymbol{w})+\lambda\|\boldsymbol{w}\|_1$

$\begin{aligned}\frac{\partial}{\partial w_d} \mathrm{NLL}(\boldsymbol{w}) & =a_d w_d-c_d \\a_d & =\sum_{n=1}^N x_{n d}^2 \\c_d & =\sum_{n=1}^N x_{n d}\left(y_n-\boldsymbol{w}_{-d}^{\top} \boldsymbol{x}_{n,-d}\right)\end{aligned}$
- $w_{-d}$ is $w$ without component $d$ and same for $x_{n,-d}$
- $\boldsymbol{r}_{-d}=\boldsymbol{y}-\mathbf{X}_{:,-d} \boldsymbol{w}_{-d}$ is the residual without component $d$
- $c_d$: This is correlation between $d$ and remaining residual
	- Large feature $d$ explains unexplained variation
- $w_d=c_d / a_d$ is the optimal update

Now adding in penalty term. We have subgradients here!

$$
\begin{aligned}
\partial_{w_d} \mathcal{L}(\boldsymbol{w}) & =\left(a_d w_d-c_d\right)+\lambda \partial_{w_d}\|\boldsymbol{w}\|_1 \\
& =\left\{\begin{array}{cl}
\left\{a_d w_d-c_d-\lambda\right\} & \text { if } w_d<0 \\
{\left[-c_d-\lambda,-c_d+\lambda\right]} & \text { if } w_d=0 \\
\left\{a_d w_d-c_d+\lambda\right\} & \text { if } w_d>0
\end{array}\right.
\end{aligned}
$$
- Can $=0$ at 3 different values of $w_d$ as follows:

1. If $c_d<-\lambda$, so the feature is strongly negatively correlated with the residual, then the subgradient is zero at $\hat{w}_d=\frac{c_d+\lambda}{a_d}<0$.
2. If $c_d \in[-\lambda, \lambda]$, so the feature is only weakly correlated with the residual, then the subgradient is zero at $\hat{w}_d=0$.
3. If $c_d>\lambda$, so the feature is strongly positively correlated with the residual, then the subgradient is zero at $\hat{w}_d=\frac{c_d-\lambda}{a_d}>0$.

Summary: $\hat{w}_d\left(c_d\right)=\left\{\begin{array}{cc}\left(c_d+\lambda\right) / a_d & \text { if } c_d<-\lambda \\0 & \text { if } c_d \in[-\lambda, \lambda] \\\left(c_d-\lambda\right) / a_d & \text { if } c_d>\lambda\end{array}\right.$
- Rewritten as : $\begin{aligned}&\hat{w}_d=\operatorname{SoftThreshold}\left(\frac{c_d}{a_d}, \lambda / a_d\right)\\&\text { where }\\&\operatorname{SoftThreshold}(x, \delta) \triangleq \operatorname{sign}(x)(|x|-\delta)_{+}\end{aligned}$
- Hard Threshold: This sets values of $w_d$ to 0 if $-\lambda \leq c_d \leq \lambda$, but it does not shrink the values of $w_d$ outside of this interval. The slope of the soft
	- Soft on the other hand, even shrinks large coefficients towards 0!
- ![[soft v hard thresholding.png]]
- Plot $\hat{w_d}$ vs $\lambda$ gives us the regularization path
	- Increasing $\lambda$ shrinks coefficients
	- That is, there are a set of critical values of λ where the active set of non-zero coefficients changes. For values of λ between these critical values, each non-zero coefficient increases or decreases in a linear fashion.

#### Lasso for Variable Selection

Goal for variable selection: With enough data, A method that can recover the true relevant variables as $N \rightarrow \infty$ is called model selection consistent.

To use Lasso for vairable selection we need to select $\lambda$
- CV
	- Means picking a value that results in good predictive accuracy
	- Not usually the same as the one to recover the true model
	- $L_1$ performs selection and shrinkage, so chosen coefficients are brought closer to 0
	- CV tends to pick a value of $\lambda$ that is not too large to prevant rlevant features from being shrunk in this way
		- Results in less sparse model which contains irrelevant features
		- because the coefficients will be highly biased, which is improved in relaxed LASSO, MCP and SCAD penalized regression, and resolved completely in L0 penalized regression (which has a full oracle property, ie it can pick out both the causal variables and return unbiased
		- 1. because it cannot deal well with collinear variables (it would essentially just randomly select one of the collinear variables) - iterative adapative ridge / `l0ara` / `L0glm` and the L0L2 penalties in `L0Learn` are much better at dealing with that. In the case of `L0glm` e.g. 2 perfectly collinear variables would end up having their effects split up equally across both.
- Shown that Prediction Optimal Value of $\lambda$ does not result in model selection consistency
	- Use relaxed LASSO, Adaptive Lasso, SCAD penalty, MCP Penalty, L0 norm, 


#### Group Lasso

Each variable may have a weight of vectors  associated with it so the overall weight vector has a block strucutre. If we want to exclude variable $d$ we have to force the whole subvector $w_d$ to zero. This is called group sparsity.

Objective: $\operatorname{PNLL}(\boldsymbol{w})=\mathrm{NLL}(\boldsymbol{w})+\lambda \sum_{g=1}^G\left\|\boldsymbol{w}_g\right\|_2$
- where $\left\|\boldsymbol{w}_q\right\|_2=\sqrt{\sum_{d \in q} w_d^2}$ is the 2 -norm of the group weight vector.
- If$NLL$ is Least Squares then this is called Group LASSO
- Squuare root enforces sparsity at the group level!
	- Alternatively replace l2 norm with infinity norm!

Inside a group coefficients are treated together and across groups sparsity is included
- Group Lasso Constraint: rounded blocks per group
- L2 does not create sparsity inside a group but allows a smooth measure of group size
	- Then oter L1-like sum across groups creates sparsity
	- L2 within groups and L1 over groups

