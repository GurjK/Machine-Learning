Regular Linear Regression does a poor job of modelling outliers
- Squared Error penalizes deviations quaratically so points far from line have more effect on fit than inside the line

To achieve robustness is to replace the Gaussian distribution with a distribution that has heavy tails
- Distribution assigns higher likllihood to outlier without have to perturb the straight line to explain them

Possible Distributions to use
- Laplace Likelihood
- Student t likelihood
- Huber Loss

Overview of Distributions:
$$
\begin{array}{llll}
\text { Likelihood } & \text { Prior } & \text { Posterior } & \text { Name } \\
\hline \text { Gaussian } & \text { Uniform } & \text { Point } & \text { Least squares } \\
\text { Student } & \text { Uniform } & \text { Point } & \text { Robust regression } \\
\text { Laplace } & \text { Uniform } & \text { Point } & \text { Robust regression } \\
\text { Gaussian } & \text { Gaussian } & \text { Point } & \text { Ridge } \\
\text { Gaussian } & \text { Laplace } & \text { Point } & \text { Lasso } \\
\text { Gaussian } & \text { Gauss-Gamma } & \text { Gauss-Gamma } & \text { Bayesian lin. reg }
\end{array}
$$


#### Laplace Distribution

Robust to Outliers

Likelihood: $p(y \mid \boldsymbol{x}, \boldsymbol{w}, b)=\operatorname{Laplace}\left(y \mid \boldsymbol{w}^{\top} \boldsymbol{x}, b\right) \propto \exp \left(-\frac{1}{b}\left|y-\boldsymbol{w}^{\top} \boldsymbol{x}\right|\right)$
- Use $|y -w^Tx|$ instead of $(y-w^Tx)^2$
- MLE is computed using Linear Programming

Linear Programming:
- $\underset{\boldsymbol{v}}{\operatorname{argmin}} \boldsymbol{c}^{\top} \boldsymbol{v}$ s.t. $\mathbf{A} \boldsymbol{v} \leq \boldsymbol{b}$
	- $v$ : unknown paramter
	- $c^Tv$: linear objective contraint
	- $a_i^Tv \leq b$ set of $m$ linear constraints

Problem
- let us define $\boldsymbol{v}=\left(w_1, \ldots, w_D, e_1, \ldots, e_N\right) \in \mathbb{R}^{D+N}$, where $e_i=\left|y_i-\hat{y}_i\right|$ is the residual error for example $i$. 

We want to minimize the sum of the residuals, so we define $\boldsymbol{c}=(0, \cdots, 0,1, \cdots, 1) \in \mathbb{R}^{D+N}$, where the first $D$ elements are 0 , and the last $N$ elements are 1 .

We need to enforce the constraint that $e_i=\left|\hat{y}_i-y_i\right|$. In fact it is sufficient to enforce the constraint that $\left|\boldsymbol{w}^{\top} \boldsymbol{x}_i-y_i\right| \leq e_i$, since minimizing the sum of the $e_i$ 's will "push down" on this constraint and make it tight. Since $|a| \leq b \Longrightarrow-b \leq a \leq b$, we can encode $\left|\boldsymbol{w}^{\top} \boldsymbol{x}_i-y_i\right| \leq e_i$ as two linear constraints:

$$
\begin{aligned}
& e_i \geq \boldsymbol{w}^{\top} \boldsymbol{x}_i-y_i \\
& e_i \geq-\left(\boldsymbol{w}^{\top} \boldsymbol{x}_i-y_i\right)
\end{aligned}
$$


We can write Equation (11.110) as

$$
\left(\boldsymbol{x}_i, 0, \cdots, 0,-1,0, \cdots, 0\right)^{\top} \boldsymbol{v} \leq y_i
$$

where the first $D$ entries are filled with $\boldsymbol{x}_i$, and the -1 is in the $(D+i)^{\prime}$ th entry of Similarly we can write Equation (11.111) as

$$
\left(-\boldsymbol{x}_i, 0, \cdots, 0,-1,0, \cdots, 0\right)^{\top} \boldsymbol{v} \leq-y_i
$$
#### Student T distribution

Make the mean be a linear function fo the inputs: $p\left(y \mid \boldsymbol{x}, \boldsymbol{w}, \sigma^2, \nu\right)=\mathcal{T}\left(y \mid \boldsymbol{w}^{\top} \boldsymbol{x}, \sigma^2, \nu\right)$
- Fit using EM or SGD

#### Huber Loss

Alternative to using NLL using student t or laplace is to use Huber Loss

$$\ell_{\text {huber }}(r, \delta)=\left\{\begin{array}{cc}
r^2 / 2 & \text { if }|r| \leq \delta \\
\delta|r|-\delta^2 / 2 & \text { if }|r|>\delta
\end{array}\right.$$
- l2 loss for errors that are smaller than $\delta$
- l1 for larger errors

Loss is differentiable everywhere
- Faster to optimize than Laplace
	- can use SGD instead of Linear Programming
	- Similar results to Laplace and Student
- $\delta$: construlls degree of robustness set by cross validation or by hand


RANSAC is another potential model
- 