
The process of estimating $\theta$ from our Data is called **model fitting or training**. This boils down to the optimization of the form $$\hat{\boldsymbol{\theta}}=\underset{\boldsymbol{\theta}}{\operatorname{argmin}} \mathcal{L}(\boldsymbol{\theta})$$
- $\mathcal{L}$ : is a loss or objective function
- $\hat{\theta}$ : point estimate
Assume we have a linear regression of the form : $p\left(y_* \mid \boldsymbol{x}_*, \boldsymbol{\theta}^*\right)=\mathcal{N}\left(y_* \mid \boldsymbol{x}_*^{\top} \boldsymbol{\theta}^*, \sigma^2\right)$. The goal is to maximize the liklihood that the data comes from the model. Put another way, maximizing the liklihood means maximizing the predictive distribution of the training data given the model parameters. We obtain the maximum liklihood parameters as

$$\hat{\boldsymbol{\theta}}_{\mathrm{mle}} \triangleq \underset{\boldsymbol{\theta}}{\operatorname{argmax}} p(\mathcal{D} \mid \boldsymbol{\theta})$$
- Typically we determind $\theta_{ML}$ by performing gradient descent
- Assume that the trainining samples are independently sampled from the same distribution so conditional liklihood is 

$$p(\mathcal{D} \mid \boldsymbol{\theta})=\prod^N p\left(\boldsymbol{y}_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}\right)$$
- This is the iid assumption

Then the likelihood of the whole dataset is:

$$
L(\theta)=\prod_{i=1}^n p\left(y_i \mid x_i, \theta\right)
$$
But to make the math more tracatable we take the Log liklihood. We take logs because products are annoying.

$$
\ell(\boldsymbol{\theta}) \triangleq \log p(\mathcal{D} \mid \boldsymbol{\theta})=\sum_{n=1}^N \log p\left(\boldsymbol{y}_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}\right)
$$

This can be decomposed as a sum therefore the MLE is given by 

$$
\hat{\boldsymbol{\theta}}_{\mathrm{mle}}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \sum_{n=1}^N \log p\left(\boldsymbol{y}_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}\right)
$$


We are currently maximizing the original function, but because the log function is montonic increasing then maximizing a quantitty is equal to maximizing its log. Maximizing a function is equal to minimizing the negative of a function. This is why we get the negative log liklihood. The negative log liklihood is given by $$
\mathrm{NLL}(\boldsymbol{\theta}) \triangleq-\log p(\mathcal{D} \mid \boldsymbol{\theta})=-\sum_{n=1}^N \log p\left(\boldsymbol{y}_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}\right)
$$
For an unsupervised model we get $$
\hat{\boldsymbol{\theta}}_{\mathrm{mle}}=\underset{\boldsymbol{\theta}}{\operatorname{argmin}}-\sum_{n=1}^N \log p\left(\boldsymbol{y}_n \mid \boldsymbol{\theta}\right)$$
### Why MLE works?

1. The predictive distribution is as close as possible to the empirical distribution of the data
2. It is a simple point approximiation to the Bayesian psoterior 

Empriical Distribution is $p_{\mathcal{D}}(\boldsymbol{y}) \triangleq \frac{1}{N} \sum_{n=1}^N \delta\left(\boldsymbol{y}-\boldsymbol{y}_n\right)$
- We want a model: $q(\boldsymbol{y})=p(\boldsymbol{y} \mid \boldsymbol{\theta})$ is similar to $p_{\mathcal{D}}(\boldsymbol{y})$
- A way to do this is to measure the Kullback Leibler divergence

Check Textbook for example of MLE for 
- Bernouli
- Categorical
- Univariate Gaussian
- Multivariate Gaussian
- Linear Regression

## Empirical Risk Minimization

Generalize MLE by replacing logg loss term with any other loss function giving us  $$\mathcal{L}(\boldsymbol{\theta})=\frac{1}{N} \sum_{n=1}^N \ell\left(\boldsymbol{y}_n, \boldsymbol{\theta} ; \boldsymbol{x}_n\right)$$
Choices for loss functiosn:
- Misclassification rate for classification problems: $$
\ell_{01}\left(\boldsymbol{y}_n, \boldsymbol{\theta} ; \boldsymbol{x}_n\right)= \begin{cases}0 & \text { if } \boldsymbol{y}_n=f\left(\boldsymbol{x}_n ; \boldsymbol{\theta}\right) \\ 1 & \text { if } \boldsymbol{y}_n \neq f\left(\boldsymbol{x}_n ; \boldsymbol{\theta}\right)\end{cases}
$$

where $f(\boldsymbol{x} ; \boldsymbol{\theta})$ is some kind of predictor. The empirical risk becomes

$$
\mathcal{L}(\boldsymbol{\theta})=\frac{1}{N} \sum_{n=1}^N \ell_{01}\left(\boldsymbol{y}_n, \boldsymbol{\theta} ; \boldsymbol{x}_n\right)
$$
- Surrogate Loss (Better than missclassifcation)

Missclassification is not smooth so difficult to run optimization on it therefore surrogate is chosen to be maximally tight convex upper bound. Let Let $\tilde{y} \in\{-1,+1\}$ be the true label, and $\hat{y} \in\{-1,+1\}=f(\boldsymbol{x} ; \boldsymbol{\theta})$ be our prediction. Then we get the following distribution For example, consider a probabilistic binary classifier, which produces the following distribution over labels:

$$
\begin{equation*}
p(\tilde{y} \mid \boldsymbol{x}, \boldsymbol{\theta})=\sigma(\tilde{y} \eta)=\frac{1}{1+e^{-\tilde{y} \eta}} 
\end{equation*}
$$

where $\eta=f(\boldsymbol{x} ; \boldsymbol{\theta})$ is the log odds. Hence the log loss is given by

$$
\begin{equation*}
\ell_{l l}(\tilde{y}, \eta)=-\log p(\tilde{y} \mid \eta)=\log \left(1+e^{-\tilde{y} \eta}\right) 
\end{equation*}
$$


Margin: $\tilde{y} \eta,$ is the true label and the prediction. 

Alternatives
- Hinge Loss: $\ell_{\text {hinge }}(\tilde{y}, \eta)=\max (0,1-\tilde{y} \eta) \triangleq(1-\tilde{y} \eta)_{+}$

