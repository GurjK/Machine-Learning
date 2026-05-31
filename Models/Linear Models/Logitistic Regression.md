**Classification Based Model**

Binary Logistic Regression: Corresponds to the following model:
$$
p(y \mid \boldsymbol{x}, \boldsymbol{\theta})=\operatorname{Ber}\left(y \mid \sigma\left(\boldsymbol{w}^{\top} \boldsymbol{x}+b\right)\right)
$$
- $\sigma$: sigmoid function
- $w$: weights
- $b$: bias
- $\theta = (w,b)$
$$
p(y=1 \mid \boldsymbol{x}, \boldsymbol{\theta})=\sigma(a)=\frac{1}{1+e^{-a}}
$$
- $a = w^tx+b = log(\frac{p}{1-p})$ and $p=p(y=1 \mid \boldsymbol{x}, \boldsymbol{\theta})$
	- $a$ = logit or the pre activation
- If we let, $y = \in (-1,1)$ then $I(a>0)$
	- $a = w^tx+b$
	- $f(\boldsymbol{x} ; \boldsymbol{\theta})=b+\boldsymbol{w}^{\top} \boldsymbol{x}=b+\sum_{d=1}^D w_d x_d$
		- $w^Tx$ is the inner product between the weight vector and feature vector
		- Function defines a linear hyperplance 
		- If we can create this decision boundary where a linear boundary makes no classification error then this is said to be linearly separable
		- $w$ defines the orientation of the decision boundary and its magnitude
- Discriminative model
- Prediction for two classes: $f(\boldsymbol{x} ; \boldsymbol{\theta})=b+\boldsymbol{w}^{\top} \boldsymbol{x}=b+\sum_{d=1}^D w_d x_d$
	- This is a hyperplane with a normal vector $w$ and offset $b$
- 



### Nonlinear Classifiers

Turn a nonlinear problem into a linear problem by preprocessing the inputs in a suitable way. 
- Let $\phi (x)$ be transformed $x$ input feature vector
- Ex: use $\boldsymbol{\phi}\left(x_1, x_2\right)=\left[1, x_1^2, x_2^2\right]$, and we let $\boldsymbol{w}=\left[-R^2, 1,1\right]$. Then $\boldsymbol{w}^{\top} \boldsymbol{\phi}(\boldsymbol{x})=x_1^2+x_2^2-R^2$, so the decision boundary (where $f(\boldsymbol{x})=0$ ) defines a circle with radius $R$, 
	- The circle decision boundary has now turned into a line!!!
	- function is still linear in parameter $w$


### MLE Estimation

Instead of classically, minimizing the average loss we are going to instead maximize the liklihood of the training data according to our mode:
- Assuming individual samples are all independent we need to find the liklihood of them by multipling the products together

Take the NLL. Remember that $\mu$ is the link function and $a_n$ is the natural parameter therefore we get and $b$ is observed into $w$ :

$$\begin{aligned}
\mathrm{NLL}(\boldsymbol{w}) & =-\frac{1}{N} \log p(\mathcal{D} \mid \boldsymbol{w})=-\frac{1}{N} \log \prod_{n=1}^N \operatorname{Ber}\left(y_n \mid \mu_n\right) \\
& =-\frac{1}{N} \sum_{n=1}^N \log \left[\mu_n^{y_n} \times\left(1-\mu_n\right)^{1-y_n}\right] \\
& =-\frac{1}{N} \sum_{n=1}^N\left[y_n \log \mu_n+\left(1-y_n\right) \log \left(1-\mu_n\right)\right] \\
& =\frac{1}{N} \sum_{n=1}^N \mathbb{H}_{c e}\left(y_n, \mu_n\right)
\end{aligned}$$
- $\mu_n = \sigma(a_n)$ is probability of class 1
- $a_n = w^t x_n$ is the logit 
- $H_{ce}(y_n,\mu_n)$ is the binary cross entropy: $\mathbb{H}_{c e}(p, q)=-[p \log q+(1-p) \log (1-q)]$

If we use $\tilde{y}_n \in\{-1,+1\}$ instead of $y_n \in\{0,1\}$, then we can rewrite this as follows:

$$
\begin{aligned}
\mathrm{NLL}(\boldsymbol{w}) & =-\frac{1}{N} \sum_{n=1}^N\left[\mathbb{I}\left(\tilde{y}_n=1\right) \log \left(\sigma\left(a_n\right)\right)+\mathbb{I}\left(\tilde{y}_n=-1\right) \log \left(\sigma\left(-a_n\right)\right)\right] \\
& =-\frac{1}{N} \sum_{n=1}^N \log \left(\sigma\left(\tilde{y}_n a_n\right)\right) \\
& =\frac{1}{N} \sum^N \log \left(1+\exp \left(-\tilde{y}_n a_n\right)\right)
\end{aligned}
$$
Gradient: $\nabla_{\boldsymbol{w}} \mathrm{NLL}(\boldsymbol{w})= \frac{1}{N} \sum_{n=1}^N\left(\mu_n-y_n\right) \boldsymbol{x}_n$
- Define error as $\mu -y$ then gradient weights each input $x_n$ by its error and averages result
Hessian:
 $$\begin{aligned}
&\mathbf{H}(\boldsymbol{w})=\nabla_{\boldsymbol{w}} \nabla_{\boldsymbol{w}}^{\top} \mathrm{NLL}(\boldsymbol{w})=\frac{1}{N} \sum_{n=1}^N\left(\mu_n\left(1-\mu_n\right) \boldsymbol{x}_n\right) \boldsymbol{x}_n^{\top}=\frac{1}{N} \mathbf{X}^{\top} \mathbf{S} \mathbf{X}\\
&\text { where }\\
&\mathbf{S} \triangleq \operatorname{diag}\left(\mu_1\left(1-\mu_1\right), \ldots, \mu_N\left(1-\mu_N\right)\right)
\end{aligned}$$
- The NLL is strictly convex but for values of $\mu$ close to 0 or 1 then it might be close to singualar. We can use l2 regularization to avoid this!!!

To find the MLE, we must solve
$$\nabla_w \mathrm{NLL}(w)=\boldsymbol{g}(\boldsymbol{w})=\mathbf{0}$$
- $\frac{1}{N} \sum_{n=1}^N\left(\mu_n-y_n\right) \boldsymbol{x}_n$ is the gradient 
- SGD: Update Equation: $\boldsymbol{w}_{t+1}=\boldsymbol{w}_t-\eta_t \nabla_{\boldsymbol{w}} \operatorname{NLL}\left(\boldsymbol{w}_t\right)=\boldsymbol{w}_t-\eta_{\mathrm{t}}\left(\mu_n-y_n\right) \boldsymbol{x}_n$


**Perceptron Algorithm**

A perceptron algortihm is a deterministic binary classifier of the form: $f\left(\boldsymbol{x}_n ; \boldsymbol{\theta}\right)=\mathbb{I}\left(\boldsymbol{w}^{\top} \boldsymbol{x}_n+b>0\right)$
- The sigmois $\sigma$ is replaced by a Heaviside step function: $H(a) \triangleq \mathbb{I}(a>0)$
- Perceptron can't be differentiated so can't use gradient descent but using the perceptron update rule we get 

Newton's Method
- Iteratively Rewuighted Least Squares:
	- At each iteration we solve a least squares problem:
	- $$
\begin{aligned}
&\boldsymbol{w}=0\\
&\text { repeat }\\
&\text { for } n=1: N \text { do }\\
&a_n=\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_n\\
&\mu_n=\sigma\left(a_n\right)\\
&s_n=\mu_n\left(1-\mu_n\right)\\
&z_n=a_n+\frac{\nu_n-\mu_n}{s_n}\\
&\mathbf{S}=\operatorname{diag}\left(s_{1: N}\right)\\
&\boldsymbol{w}=\left(\mathbf{X}^{\boldsymbol{\top}} \mathbf{S X}\right)^{-1} \mathbf{X}^{\boldsymbol{\top}} \mathbf{S} \boldsymbol{z}\\
&\text { until converged }
\end{aligned}
$$
### MAP Estimation

Issues with Logistic Regression with MLE
- Can overfit when there are too many parameters compared to training examples
- Consequence of ability of maximum likelihood to find weights that foce the decision boundary to "wiggle" so curve gets around examples

Solution:
- Use a zero mean Gaussian prior: $p(\boldsymbol{w})=\mathcal{N}(\boldsymbol{w} \mid \mathbf{0}, C \mathbf{I})$,
- Then MAP Estimation: $\mathcal{L}(\boldsymbol{w})=\operatorname{NLL}(\boldsymbol{w})+\lambda\|\boldsymbol{w}\|_2^2$
	- $\lambda = 1/C$
	- $L2$ Regulatization or Weight Decay
- This gives Penalized Negative Log Liklihood
	- $\begin{aligned} \operatorname{PNLL}(\boldsymbol{w}) & =\mathrm{NLL}(\boldsymbol{w})+\lambda \boldsymbol{w}^{\top} \boldsymbol{w} \\ \nabla_{\boldsymbol{w}} \operatorname{PNLL}(\boldsymbol{w}) & =\boldsymbol{g}(\boldsymbol{w})+2 \lambda \boldsymbol{w} \\ \nabla_{\boldsymbol{w}}^2 \operatorname{PNLL}(\boldsymbol{w}) & =\mathbf{H}(\boldsymbol{w})+2 \lambda \mathbf{I}\end{aligned}$
		- $H$ and $g$ are Hessian and gradient of unpenalized NLL
- Preprocessing needed since we require all weights to be similar in magnitude
	- Standardization: $\operatorname{standardize}\left(x_{n d}\right)=\frac{x_{n d}-\hat{\mu}_d}{\hat{\sigma}_d}$
	- Min Max Scaling: Change to lie between 0 and 1


### Multinomial Logistic Regression

Discriminative Classification of the form: $p(y \mid \boldsymbol{x}, \boldsymbol{\theta})=\operatorname{Cat}(y \mid \operatorname{softmax}(\mathbf{W} \boldsymbol{x}+\boldsymbol{b}))$
- where $\boldsymbol{x} \in \mathbb{R}^D$ is the input vector,
- $y \in\{1, \ldots, C\}$ is the class label,
- softmax() is the softmax function: $\sigma(\mathbf{z})_i=\frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$
- $\mathbf{W}$ is a $C \times D$ weight matrix,
- $\boldsymbol{b}$ is $C$-dimensional bias vector
- , $\boldsymbol{\theta}=(\mathbf{W}, \boldsymbol{b})$ are all the parameters. (We will henceforth assume we have prepended each $\boldsymbol{x}$ with a 1 , and added $\boldsymbol{b}$ to the first column of $\mathbf{W}$, so this simplifies to $\boldsymbol{\theta}=\mathbf{W}$.)
- Let $a = Wx$ be C- dimensiona vector of logits then we get:
$$
p(y=c \mid \boldsymbol{x}, \boldsymbol{\theta})=\frac{e^{a_c}}{\sum_{c^{\prime}=1}^C e^{a_{c^{\prime}}}}
$$

NLL: $\mathrm{NLL}(\boldsymbol{\theta})=-\frac{1}{N} \log \prod_{n=1}^N \prod_{c=1}^C \mu_{n c}^{y_{n c}}=-\frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C y_{n c} \log \mu_{n c}=\frac{1}{N} \sum_{n=1}^N \mathbb{H}_{c e}\left(\boldsymbol{y}_n, \boldsymbol{\mu}_n\right)$
- $\mathbb{H}_{c e}(\boldsymbol{p}, \boldsymbol{q})=-\sum_{c=1}^C p_c \log q_c$
- where $\mu_{n c}=p\left(y_{n c}=1 \mid \boldsymbol{x}_n, \boldsymbol{\theta}\right)=\operatorname{softmax}\left(f\left(\boldsymbol{x}_n, \boldsymbol{\theta}\right)\right)_c,$ 
- $\boldsymbol{y}_n$ is the one-hot encoding of the label (so $\left.y_{n c}=\mathbb{I}\left(y_n=c\right)\right)$, 
- $\mathbb{H}_{c e}\left(\boldsymbol{y}_n, \boldsymbol{\mu}_n\right)$ is the cross-entropy: $\mathbb{H}_{c e}(\boldsymbol{p}, \boldsymbol{q})=-\sum_{c=1}^C p_c \log q_c$
Gradient: $\boldsymbol{g}(\boldsymbol{w})=\frac{1}{N} \sum_{n=1}^N \boldsymbol{x}_n\left(\boldsymbol{\mu}_n-\boldsymbol{y}_n\right)^{\top}$
Hessian: $=\frac{1}{N} \sum_n\left(\begin{array}{cc}\left(\mu_{n 1}-\mu_{n 1}^2\right) \mathbf{X}_n & -\mu_{n 1} \mu_{n 2} \mathbf{X}_n \\ -\mu_{n 1} \mu_{n 2} \mathbf{X}_n & \left(\mu_{n 2}-\mu_{n 2}^2\right) \mathbf{X}_n\end{array}\right)$

##### MAP Estimation

Issue in multiclass is that many different parameter values can give the same model.
- There is no unique optimum in optimization problems
- Many parameter choices give the same liklihood
Solution
- Set one weights class to 0
- L2 regularization fixes this naturally
	- Optimizer perfers one with smallest norm!
	- - removes non-identifiability and gives a unique solution
	- $\operatorname{PNLL}(\mathbf{W})=-\sum_{n=1}^N \log p\left(y_n \mid \boldsymbol{x}_n, \mathbf{W}\right)+\lambda \sum_{c=1}^C\left\|\boldsymbol{w}_c\right\|_2^2$

**Maximum Entropy Classifier**

$p(y=c \mid \boldsymbol{x}, \boldsymbol{w})=\frac{1}{Z(\boldsymbol{w}, \boldsymbol{x})} \exp \left(\boldsymbol{w}^{\top} \phi(\boldsymbol{x}, c)\right)$
- $Z(\boldsymbol{w}, \boldsymbol{x})=\sum \exp \left(\boldsymbol{w}_c^{\top} \boldsymbol{x}\right)$ is the partition function (normalization constant).
- $\phi(\boldsymbol{x}, c)$ is the feature vector for class $c$.

Solutions to handling a large number of classes
- Hierachical softmax
- Class Imbalance and long tail: 
	- logit adjustment
	- instance-balanced sampling
	- class balanced sampling
	- nearest mean class classifier



