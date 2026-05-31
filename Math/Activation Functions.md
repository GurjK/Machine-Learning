## Sigmoid Function

Consider needing to predict a binary variable in the range $y \in (0,1)$ given inputs $x$ then we need to use a conditional probability of the form $p(y \mid \boldsymbol{x}, \boldsymbol{\theta})=\operatorname{Ber}(y \mid f(\boldsymbol{x} ; \boldsymbol{\theta}))$

- $f(x;\theta)$: function that predicts the mean parameter of the output distribution
	- Let $f$ be an unconstrained function then we can use the following model
	- $x$ are the input features or the predictors
	- $\theta$: The model weights or parameters of the model
		- Ex: $f(x,\theta) = \theta^T x +b$

$$
p(y \mid \boldsymbol{x}, \boldsymbol{\theta})=\operatorname{Ber}(y \mid \sigma(f(\boldsymbol{x} ; \boldsymbol{\theta})))
$$
- $\sigma(a) \triangleq \frac{1}{1+e^{-a}}$
	- $a = f(x;\theta)$

![[sigmoid.png]]

So the entire flow is 

$x \rightarrow f(x ; \theta) \rightarrow \sigma(f(x ; \theta)) \rightarrow P(y=1 \mid x, \theta)$

In Bernouli Setup y can only be 1 or 0 which is why we can model this as 
$$
p(y \mid x, \theta)=\operatorname{Ber}(y \mid \sigma(f(x ; \theta)))
$$
meaning:
$$
y \sim \operatorname{Bernoulli}(p) \text { with } p=\sigma(f(x ; \theta))
$$
- $p$ is log odds here : $\log \left(\frac{p}{1-p}\right)=\log \left(\frac{e^a}{1+e^a} \frac{1+e^a}{1}\right)=\log \left(e^a\right)=a$

The logistic function or sigmoid function maps the log-odds $a$ to $p$ :

$$
p=\operatorname{logistic}(a)=\sigma(a) \triangleq \frac{1}{1+e^{-a}}=\frac{e^a}{1+e^a}
$$


The inverse of this is called the logit function, and maps $p$ to the log-odds $a$ :

$$
a=\operatorname{logit}(p)=\sigma^{-1}(p) \triangleq \log \left(\frac{p}{1-p}\right)
$$

Interpretation of coefficients:
Suppose:

$$
\log \left(\frac{p}{1-p}\right)=\theta_0+\theta_1 x_1
$$


Then:
- increasing $x_1$ by 1 changes the log-odds by $\theta_1$
- increasing $x_1$ by 1 multiplies the odds by:

## Softmax Function

 A categorical distribution in the conditional case can be written as 
 In the conditional case, we can define

$$
p(y \mid \boldsymbol{x}, \boldsymbol{\theta})=\operatorname{Cat}(y \mid f(\boldsymbol{x} ; \boldsymbol{\theta}))
$$

which we can also write as

$$
p(y \mid \boldsymbol{x}, \boldsymbol{\theta})=\mathcal{M}(\boldsymbol{y} \mid 1, f(\boldsymbol{x} ; \boldsymbol{\theta}))
$$


We require that $0 \leq f_c(\boldsymbol{x} ; \boldsymbol{\theta}) \leq 1$ and $\sum_{c=1}^C f_c(\boldsymbol{x} ; \boldsymbol{\theta})=1$.

To avoid requirement that $f$ is a probability vector pass $f$ into a **softmax** function aslo called a multinomial logit defined as 

$$\operatorname{softmax}(\boldsymbol{a}) \triangleq\left[\frac{e^{a_1}}{\sum_{c^{\prime}=1}^C e^{a_{c^{\prime}}}}, \ldots, \frac{e^{a_C}}{\sum_{c^{\prime}=1}^C e^{a_{c^{\prime}}}}\right]$$
- Maps: $R^C \to [0,1]^C$
- $\sum \text{softmax (a)} = 1$

Used in Multiclass Logisitic Regression!
## Softplus Function

