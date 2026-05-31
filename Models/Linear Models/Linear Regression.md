
**Regression Based Model**
### Univariate Linear Regression

$Y$: Predicting variable
$X$: single predictor variable
$\beta$: unknown coefficients

The data consists of $\left(x_1, y_1\right),\left(x_2, y_2\right), \ldots,\left(x_n, y_n\right)$ . The goal is to estimate the $\beta$ such that we generate $y_i \approx \hat{\beta}_0+\hat{\beta}_1 x_i \text { for } i=1, \ldots, n$.


$$\hat{y} = \hat{\beta_0} + \hat{\beta_1}x$$
How to determine the minimum amount: Sum of Squares Residual (RSS). The solution in the univariate setting is 
$\begin{aligned} & \hat{\beta}_1=\frac{\sum_{i=1}^n\left(x_i-\bar{x}\right)\left(y_i-\bar{y}\right)}{\sum_{i=1}^n\left(x_i-\bar{x}\right)^2} = \frac{\text{Cov(X,Y)}}{Var{(X)}}, \\ & \hat{\beta}_0=\bar{y}-\hat{\beta}_1 \bar{x},\end{aligned}$

or 

Let $e_i=y_i-\hat{y}_i$ represent the $i$th residual, this is the difference between the ith observed value and the ith predicted value. Then we get that the 


Random Questions:
Is at least one of the predictors X1,X2,...,Xp useful in predicting the response?
- Use F Statistic to determine if at least $\beta_j$ is nonzero. Hypothesis Test
Do all the predictors help to explain Y , or is only a subset of the predictors useful?
- Use Forward Selection, Backward, or Mixed
- Forward is greedy tho and might result in redundant variables

How well does the model fit the data?
- RSE and R-Squared

### How to Extend a simple Linear Model?

1) Remove the additive assumption. Include an Interaction Effect

Consider the standard linear regression model with two variables,

$$
Y=\beta_0+\beta_1 X_1+\beta_2 X_2+\epsilon
$$


According to this model, a one-unit increase in $X_1$ is associated with an average increase in $Y$ of $\beta_1$ units. Notice that the presence of $X_2$ does not alter this statement - that is, regardless of the value of $X_2$, a oneunit increase in $X_1$ is associated with a $\beta_1$-unit increase in $Y$. One way of extending this model is to include a third predictor, called an interaction term, which is constructed by computing the product of $X_1$ and $X_2$. This results in the model

$$Y=\beta_0+\beta_1 X_1+\beta_2 X_2+\beta_3 X_1 X_2+\epsilon$$
Now the equation relaxes the additivity since we can rewrite this as!

$$\begin{aligned}
Y & =\beta_0+\left(\beta_1+\beta_3 X_2\right) X_1+\beta_2 X_2+\epsilon \\
& =\beta_0+\tilde{\beta}_1 X_1+\beta_2 X_2+\epsilon
\end{aligned}$$
- The hierarchical principle states that if we include an interaction in a model, we hierarchical principle  should also include the main effects, even if the p-values associated with their coefficients are not significant.

2) Non-Linear Relationships

We can square a Feature!! Keeps the model to be Linear



### Problems in Linear Regression


1. Non-linearity of the response-predictor relationships.
	1. Use Residual Plots to identify non-linearity. Plot residuals versus fitted values. There should ideally be no discernable pattern
	2. I.e. a strong U shape in residuals means some non linearity present!
		1. An approach to handle this could be to introduce log(X), sqrt(X), X^2
2. Correlation of error terms.
	1. If in fact there is correlation among the error terms, then the estimated standard errors will tend to underestimate the true standard errors.
		1. Your Statistics will be underestimating the true values!
	2. Plot residuals of observations over time. And check for their correlations
3. Non-constant variance of error terms.
	1. One can identify non-constant variances in the errors, or heteroscedasticity, from the presence of a funnel shape in hetero the residual plot.
	2. Transform the response $Y$ by taking log(Y) or sqrt(Y)
4. Outliers.
	1. Might not effect the fit but could have other issues. RSE might be alot higher
	2. Take studentized residuals and if greater than 3 then considered outliers
	3. May indicate definicieny in model
5. High-leverage points.
	1. If Regression is affected massively by just a couple of points
	2. Compute the leverage statistic: $h_i=\frac{1}{n}+\frac{\left(x_i-\bar{x}\right)^2}{\sum_{i^{\prime}-1}^n\left(x_{i^{\prime}}-\bar{x}\right)^2}$.
		1. 
6. Collinearity.
	1. When to features move together!
	2. Reduces accuracy of the regression coefficeints, caused SE of $\beta_j$ to grow. Results in decline for the t statistic. May fail to reject null hypothesis in this case
	3. Identify by looking at the correlation matrix!
		1. it is possible for collinear ity to exist between three or more variables even if no pair of variables has a particularly high correlation. We call this situation multicollinearity.
		2. Compute the Variance Inflation Factor (VIF): 
			1. 1 means no collienarity. > 5/10m indicates collinearity
			2. $\operatorname{VIF}\left(\hat{\beta}_j\right)=\frac{1}{1-R_{X_j \mid X_{-j}}^2}$
	4. Solutions
		1. Drop the problematic variable from regression
		2. Combine collinear variables into single variable. Take avareage of standard deviation for example!



## Bayesian Intepretation of Regression

suppose we want to measure this in bayesian terms then 

$p(y \mid \boldsymbol{x} ; \boldsymbol{\theta})=\mathcal{N}\left(y \mid f_\mu(\boldsymbol{x} ; \boldsymbol{\theta}), f_\sigma(\boldsymbol{x} ; \boldsymbol{\theta})^2\right)$
- $f_u(\boldsymbol{x} ; \boldsymbol{\theta}) \in \mathbb{R}$ predicts the mean, and $f_\sigma(\boldsymbol{x} ; \boldsymbol{\theta})^2 \in \mathbb{R}_{+}$predicts the variance.

If we assume $f_u$ to be linear then we get a **Linear Regression** :

$p(y \mid \boldsymbol{x} ; \boldsymbol{\theta})=\mathcal{N}\left(y \mid \boldsymbol{w}^{\top} \boldsymbol{x}+b, \sigma^2\right)$
- $\theta = (w,b,\sigma^2)$
- Homoskedastic regression where we assume a fixed variance

Suppose we want to make the variance depend on the input then this would be a heteroskedastic regression where


$$
p(y \mid \boldsymbol{x} ; \boldsymbol{\theta})=\mathcal{N}\left(y \mid \boldsymbol{w}_\mu^{\top} \boldsymbol{x}+b, \sigma_{+}\left(\boldsymbol{w}_\sigma^{\top} \boldsymbol{x}\right)\right)
$$

where $\boldsymbol{\theta}=\left(\boldsymbol{w}_\mu, \boldsymbol{w}_\sigma\right)$ are the two forms of regression weights, and

$$
\sigma_{+}(a)=\log \left(1+e^a\right)
$$
- This is called a **softplus** function
	- Ensures that the variance will always be positive


Assume that we wish to use the classic version with a constant included and constant variance then we have 
$$
p(y \mid \boldsymbol{x}, \boldsymbol{\theta})=\mathcal{N}\left(y \mid w_0+\boldsymbol{w}^{\top} \boldsymbol{x}, \sigma^2\right)
$$
here $\boldsymbol{\theta}=\left(w_0, \boldsymbol{w}, \sigma^2\right)$ are all the parameters of the model.
- $w_0$: uncoditional mean of the data
- $w$: weights/coefficients of the features

Suppose we assume that a linear regression is not complex enough then we can apply non-linear transformation to $x$: 

$p(y \mid \boldsymbol{x}, \boldsymbol{\theta})=\mathcal{N}\left(y \mid \boldsymbol{w}^{\top} \boldsymbol{\phi}(\boldsymbol{x}), \sigma^2\right)$
- $\phi(x)$: non-linear transformation applied
	- $\phi(x)=\left[1, x, x^2, \ldots, x^d\right]$ called the Feature Extractor


#### Solving the OLS Equation

First to fit linear regression model to data we minimize the negative log likelihood on the training set. The objective is given by $$\begin{aligned}
\mathrm{NLL}\left(\boldsymbol{w}, \sigma^2\right) & =-\sum_{n=1}^N \log \left[\left(\frac{1}{2 \pi \sigma^2}\right)^{\frac{1}{2}} \exp \left(-\frac{1}{2 \sigma^2}\left(y_n-\boldsymbol{w}^{\top} \boldsymbol{x}_n\right)^2\right)\right] \\
& =\frac{1}{2 \sigma^2} \sum_{n=1}^N\left(y_n-\hat{y}_n\right)^2+\frac{N}{2} \log \left(2 \pi \sigma^2\right)
\end{aligned}$$
- Second term is a constant so it doesn't matter
- predicted response: $\hat{y}_n \triangleq \boldsymbol{w}^{\top} \boldsymbol{x}_n$
- This is just the residual sum of squares!!!!

MLE is the point where the gradient of the NLL=0 or where $\nabla_{\boldsymbol{w}, \sigma} \mathrm{NLL}\left(\boldsymbol{w}, \sigma^2\right)=0$. So first optimize wrt to $w$ and the solve for the optimal $\sigma$. Solving for the $w$ is equal to the residual sum of squares which is given by $$\operatorname{RSS}(\boldsymbol{w})=\frac{1}{2} \sum_{n=1}^N\left(y_n-\boldsymbol{w}^{\top} \boldsymbol{x}_n\right)^2=\frac{1}{2}\|\mathbf{X} \boldsymbol{w}-\boldsymbol{y}\|_2^2=\frac{1}{2}(\mathbf{X} \boldsymbol{w}-\boldsymbol{y})^{\top}(\mathbf{X} \boldsymbol{w}-\boldsymbol{y})$$
- This 
Goal is to minimize the least squared cost function which to get as close as possible to solving $Ax=b$ exactly:
$$f(\boldsymbol{w})=\frac{1}{2}\|\mathbf{X} \boldsymbol{w}-\boldsymbol{y}\|_2^2$$
First we can recgonize to solve this we must compute the gradient using matrix calculus:
$$\boldsymbol{g}(\boldsymbol{w})=\frac{\partial}{\partial \boldsymbol{w}} f(\boldsymbol{w})=\mathbf{X}^{\top} \mathbf{X} \boldsymbol{w}-\mathbf{X}^{\top} \boldsymbol{y}$$
- The optimal solution is found by setting the $g(x) =0$

This gives: $$\mathbf{X}^{\top} \mathbf{X} \boldsymbol{w}=\mathbf{X}^{\top} \boldsymbol{y}$$
- These are known as the normal equations since at optimal solution, $b-Ax$ is normal to the range of $A$

The solution by solving for $\hat{x}$ is : $$\hat{\boldsymbol{w}}=\left(\mathbf{X}^{\top} \mathbf{X}\right)^{-1} \mathbf{X}^{\top} \boldsymbol{y}$$
We can check solutions is unique, by computing the Hessian which is $$\mathbf{H}(\boldsymbol{x})=\frac{\partial^2}{\partial \boldsymbol{x}^2} f(\boldsymbol{x})=\mathbf{X}^{\top} \mathbf{X}$$
- As long as $A$ is full rank then we get that $H4 is positive definite
- Normal Equation: $\mathbf{X}^{\top} \mathbf{X} \boldsymbol{w}=\mathbf{X}^{\top} \boldsymbol{y}$
	- This is normal( orthogonal) to the range of $X$

The noise variance is given by : $$\hat{\sigma}_{\mathrm{mle}}^2=\underset{\sigma^2}{\operatorname{argmin}} \mathrm{NLL}\left(\hat{\boldsymbol{w}}, \sigma^2\right)=\frac{1}{N} \sum_{n=1}^N\left(y_n-\boldsymbol{x}_n^{\top} \hat{\boldsymbol{w}}\right)^2$$

#### Geometric Interpretation of Least Squares

Assume $N > D$, more observations then unknowns. Goal is to find  a vector $\hat{\boldsymbol{y}} \in \mathbb{R}^N$ that lies in the linear subspace spanned by $\mathbf{X}$ and is as close as possible to $\boldsymbol{y}$, i.e., we want to find
$\underset{\hat{\boldsymbol{y}} \in \operatorname{span}\left(\left\{\boldsymbol{x}_{i, 1}, \ldots, \boldsymbol{x}_{i, d}\right\}\right)}{\operatorname{argmin}}\|\boldsymbol{y}-\hat{\boldsymbol{y}}\|_2$.. Since $\hat{y}$ lies in the linear subspace of $X$ then there exists a $w$ such that $\hat{\boldsymbol{y}}=w_1 \boldsymbol{x}_{:, 1}+\cdots+w_D \boldsymbol{x}_{:, D}=\mathbf{X} \boldsymbol{w}$. To minimize the norm of the residual, $y -\hat{y}$ we want residual vector to be orthogonal to every column of $X$ so $$\boldsymbol{x}_{:, d}^{\boldsymbol{\top}}(\boldsymbol{y}-\hat{\boldsymbol{y}})=0 \Rightarrow \mathbf{X}^{\boldsymbol{\top}}(\boldsymbol{y}-\mathbf{X} \boldsymbol{w})=\mathbf{0} \Rightarrow \boldsymbol{w}=\left(\mathbf{X}^{\boldsymbol{\top}} \mathbf{X}\right)^{-1} \mathbf{X}^{\boldsymbol{\top}} \boldsymbol{y}$$
Therefore the projective value of $y$ is given by $$\hat{\boldsymbol{y}}=\mathbf{X} \boldsymbol{w}=\mathbf{X}\left(\mathbf{X}^{\top} \mathbf{X}\right)^{-1} \mathbf{X}^{\top} \boldsymbol{y}$$
- This is the orthogonal projection of $y$ onto column space of $X$

The projection matrix is $$\operatorname{Proj}(\mathbf{X}) \triangleq \mathbf{X}\left(\mathbf{X}^{\top} \mathbf{X}\right)^{-1} \mathbf{X}^{\top}$$
is called the hat matrix since $\hat{y} = Proj(X)y$. If $X=x$ is a column vector then , the orhthongal projection of $y$ onto line $x$ is $\operatorname{Proj}(x) y=x \frac{x^{\top} y}{x^{\top} x}$


#### Algorithmic Issues

Never actually invert this matrix $X^TX$ because it may be numerically unstable. It is better to using the SVD. 
- SKLearn uses: the scipy.linalg.lstsq function, which in turns calls DGELSD, which is an SVD-based solver implemented by the LAPACK library, written in Fortran.

If $X$ is tall and skinny then quicker to us $QR$ decomposition.
- $X=QR$, $Q^TQ=I$
- $\begin{aligned}(\mathbf{Q R}) \boldsymbol{w} & =\boldsymbol{y} \\ \mathbf{Q}^{\top} \mathbf{Q R} \boldsymbol{w} & =\mathbf{Q}^{\top} \boldsymbol{y} \\ \boldsymbol{w} & =\mathbf{R}^{-1}\left(\mathbf{Q}^{\top} \boldsymbol{y}\right)\end{aligned}$
	- $R$ is upper triangular

Instead of Direct methods like QR and SVD can use Iterative MEthods
- Conjugate Gradient 
	- Assumes: $X$ symmetric
- GMRES
	- Assumes: Any $X$:
	- sparse.linalg.gmres
- Well suited for $X$ is sparse or structured

**Make sure to standardize the input features before fitting to ensure zero mean and unit variance**

#### Weighted Least Squares

This is for when we assume that the variance depends on the input resulting in a **Heteroskedastic Regression**
$$
p(y \mid \boldsymbol{x} ; \boldsymbol{\theta})=\mathcal{N}\left(y \mid \boldsymbol{w}^{\top} \boldsymbol{x}, \sigma^2(\boldsymbol{x})\right)=\frac{1}{\sqrt{2 \pi \sigma^2(\boldsymbol{x})}} \exp \left(-\frac{1}{2 \sigma^2(\boldsymbol{x})}\left(y-\boldsymbol{w}^{\top} \boldsymbol{x}\right)^2\right)
$$
Thus
$$p(\boldsymbol{y} \mid \boldsymbol{x} ; \boldsymbol{\theta})=\mathcal{N}\left(\boldsymbol{y} \mid \mathbf{X} \boldsymbol{w}, \mathbf{\Lambda}^{-1}\right)$$
where $\boldsymbol{\Lambda}=\operatorname{diag}\left(1 / \sigma^2\left(\boldsymbol{x}_n\right)\right)$. This is known as weighted linear regression. One can show that the MLE is given by$$\hat{\boldsymbol{w}}=\left(\mathbf{X}^{\top} \boldsymbol{\Lambda} \mathbf{X}\right)^{-1} \mathbf{X}^{\top} \boldsymbol{\Lambda} \boldsymbol{y}$$
This is known as the weighted least squares estimate.

#### Interpretation

Assume 1d inputs then, we can show that the solution is

$$\begin{aligned}
& \hat{w}_1=\frac{\sum_n\left(x_n-\bar{x}\right)\left(y_n-\bar{y}\right)}{\sum_n\left(x_n-\bar{x}\right)^2}=\frac{C_{x y}}{C_{x x}}=\rho_{x y} \frac{\sigma_y}{\sigma_x} \\
& \hat{w}_0=\mathbb{E}[y]-w_1 \mathbb{E}[x] \approx \bar{y}-\hat{w}_1 \bar{x}
\end{aligned}$$
where $C_{x x}=\operatorname{Cov}[X, X]=\mathbb{V}[X]=\sigma_x^2, C_{y y}=\operatorname{Cov}[Y, Y]=\mathbb{V}[Y]=\sigma_y^2, C_{x y}=\operatorname{Cov}[X, Y]$, and $\rho_{x y}=\frac{C_{x y}}{\sigma_x \sigma_y}$. Hence the prediction becomes $\hat{y}=\bar{y}-\hat{w}_1 \bar{x}+\hat{w}_1 x=\bar{y}=\rho \sigma_y \frac{(x-\bar{x})}{\sigma_y}$. We can interpret this equation as follows: we estimate how many standard deviations $x$ is away from the mean of $X$, and then we predict the outcome to be the mean of $Y$ plus $\rho$ times that many standard deviations above or below.
- Regression Coefficient: $\triangleq \frac{\partial}{\partial x} \mathbb{E}[Y \mid X=x]=w_1=\frac{C_{x y}}{C_{x x}}$
	- Slope of linear prediction for $Y$ given $X$
- Partial Regression Coefficient: $w_1=R_{Y X_1 \cdot X_2}=\frac{\partial}{\partial x} \mathbb{E}\left[Y \mid X_1=x, X_2\right]$

#### Least Mean Squares

While OLS is the exact solution to the linear regression.
- Uses all the data at once
- Gives exact minimizer
- Common in stats

LMS is iterative adaptive algorithm, instead of solving exactly it updates parameters using gradient information from one sample at a time.

Update rule:
$$
w_{t+1}=w_t+\eta e_t x_t
$$
where
$$
e_t=y_t-x_t^{\top} w_t
$$
This is essentially SGD on squared loss. 


#### Linear Regression Goodness of Fit

**Residual Plots**
For 1d inputs: plot the residuals, $r_n = y_n -\hat{y_n}$ vs $x_n$
- Model assumes: $\mathcal{N}\left(0, \sigma^2\right)$
	- So residual plot should be a cloud of point equally above or below the horizontal line at 0 without trends
	- Curvature indicates lack of fit
For Multidimensional inputs
- Plot $\hat{y_n}$ vs $y_n$
- A good model will have points that lie on a  diagonal line!

**Prediction Accuracy and $R^2$**

Quantitative compute the following:
- Residual Sum of Squares: $\operatorname{RSS}(\boldsymbol{w})=\sum_{n=1}^N\left(y_n-\boldsymbol{w}^{\top} \boldsymbol{x}_n\right)^2$
- Root Mean Squared Error: $\operatorname{RMSE}(\boldsymbol{w}) \triangleq \sqrt{\frac{1}{N} \operatorname{RSS}(\boldsymbol{w})}$
- Coefficient of Determination: $R^2 \triangleq 1-\frac{\sum_{n=1}^N\left(\hat{y}_n-y_n\right)^2}{\sum_{n=1}^N\left(\bar{y}-y_n\right)^2}=1-\frac{\mathrm{RSS}}{\mathrm{TSS}}$
	- $\bar{y}=\frac{1}{N} \sum_{n=1}^N y_n$ is the empirical mean of the response
	- $\mathrm{RSS}=\sum_{n=1}^N\left(y_n-\hat{y}_n\right)^2$ is Residual Sum of Squares
	- $\mathrm{TSS}=\sum_{n=1}^N\left(y_n-\bar{y}\right)^2$ is the total sum of squares
	- Measures the variance in the predictions relative to a simple predicative of the mean
		- No better than mean than $R^2=0$
		- Perfect Fit: $R^2=1$

**It is very common to model the noise in regression models using a Gaussian distribution with zero mean and constant variance, $r_n \sim \mathcal{N}\left(0, \sigma^2\right)$, where $r_n=y_n-\boldsymbol{w}^{\top} \boldsymbol{x}_n$. In this case, maximizing likelihood is equivalent to minimizing the sum of squared residuals, as we have seen.**

