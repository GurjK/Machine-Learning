Conditionally Independent: Two variables, X and Y, are conditionally independent given Z if the conditional joint can be written as a product of conditional marginals as $X \perp Y \mid Z \Longleftrightarrow p(X, Y \mid Z)=p(X \mid Z) p(Y \mid Z)$

#### Bayesian Rule

$$p(H=h \mid Y=y)=\frac{p(H=h) p(Y=y \mid H=h)}{p(Y=y)}$$
- $H$ is an unknown quantity
- $p(H)$: prior distribution
- $p(Y=y \mid H=h)$: liklihood or posterior distriution
- $p(Y|H=h)$: observation distribution
- $p(Y=y)$: marginal liklihood

Bayes Rule can be summarized as Posterio is proportional to prior * liklelihood

Statistics
- Sensitivity: $p(Y=1) | (H=1)$
- False Negative: 1 - Sensitivity
- Specificity (True Negative Rate): $p(Y=0) | (H=0)$
- False Positive: 1- Specificity


Inverse Probability
- concerned with inferring state of world from observations of outcomes
- Use Bayes to compute Posterior, requires quantifying prior and 

# Bayesian Statistics


$$p(\boldsymbol{\theta} \mid \mathcal{D})=\frac{p(\boldsymbol{\theta}) p(\mathcal{D} \mid \boldsymbol{\theta})}{p(\mathcal{D})}=\frac{p(\boldsymbol{\theta}) p(\mathcal{D} \mid \boldsymbol{\theta})}{\int p\left(\boldsymbol{\theta}^{\prime}\right) p\left(\mathcal{D} \mid \boldsymbol{\theta}^{\prime}\right) d \boldsymbol{\theta}^{\prime}}$$
- $p(\boldsymbol{\theta})$: Prior Distribution. What we know before seeing the data
- $p(\mathcal{D} \mid \boldsymbol{\theta})$: Liklihood Function: Te data we expect to see for each setting of the parameters
- $p(\mathcal{D})$: Marginal Liklihood: Average probability of the data where the average wrt the prior. 
- $p(\boldsymbol{\theta} \mid \mathcal{D})$: Posterior

Once we have computed the posterior over the parameters, we can compute the posterior predictive distribution over outputs given inputs by marginalizing out the unknown parameters. In the supervised/ conditional case, this becomes

$$
\begin{equation*}
p(\boldsymbol{y} \mid \boldsymbol{x}, \mathcal{D})=\int p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta} 
\end{equation*}
$$


This can be viewed as a form of Bayes model averaging (BMA), since we are making predictions using an infinite set of models (parameter values), each one weighted by how likely it is. The use of BMA reduces the chance of overfitting (Section 1.2.3), since we are not just using the single best model.


### Beta Binomial Model

Toss a coin $N$ times, and want to determine the probability of heads. Let $y_n = 1$ denote the event that the nth trial was heads. $y_n$ is the event that the nth trial was tails. 

You have 2 options
- Bernouli Liklihood with events that are iid: $p(\mathcal{D} \mid \theta)=\prod_{n=1}^N \theta^{y_n}(1-\theta)^{1-y_n}=\theta^{N_1}(1-\theta)^{N_0}$
- Binomial Liklihood: $p(\mathcal{D} \mid \theta)=\operatorname{Bin}(y \mid N, \theta)=\binom{N}{y} \theta^y(1-\theta)^{N-y}$

We need to specify a distribution about our beliefs before seeing the data. The Beta distribution is a conjugate pior for the bernouli and binmoial liklihoods while means the resulting posterior distribution will also be a beta distribution.

$$p(\theta) \propto \theta^{\breve{\alpha}-1}(1-\theta)^{\breve{\beta}-1} \propto \operatorname{Beta}(\theta \mid \breve{\alpha}, \breve{\beta})$$
Then we can Multiply the Bernouli with the beta prior to get a beta posterior as followws: $$\begin{aligned}
&\begin{aligned}
p(\theta \mid \mathcal{D}) & \propto \theta^{N_1}(1-\theta)^{N_0} \theta^{\breve{\alpha}-1}(1-\theta)^{\breve{\beta}-1} \\
& \propto \operatorname{Beta}\left(\theta \mid \breve{\alpha}+N_1, \breve{\beta}+N_0\right) \\
& =\operatorname{Beta}(\theta \mid \overparen{\alpha}, \overparen{\beta})
\end{aligned}\\
&\text { where } \widehat{\alpha} \triangleq \breve{\alpha}+N_1 \text { and } \widehat{\beta} \triangleq \breve{\beta}+N_0 \text { are the parameters of the posterior. }
\end{aligned}$$
- The terms in the prior are called hyperparameters
- Ex: Using Beta(2,2) results in a posterior becoming Beta(6,3)
- Beta(1,1) results in a unform(0,1) so a straight line at 1

- Map Estimate: Single most probable value for $\theta$ in the posterior distribution
	- This is the mode of the posterior distribution! Hence Maximum a Posterior
- Posterior Mean: weighted average of the prior mean and the MLE
- 

So far, for MLE and MoM estimation, we assumed $\theta$ was fixed but unknown. This is called the Frequentist framework where we only estimate our parameter based on data alone, and $\theta$ is not a random variable. Now, we are in the Bayesian framework, meaning that our unknown parameter is a random variable $\Theta$. This means, we will have some belief distribution $\pi_{\Theta}(\theta)$ (think of this as a density function over all possible values of the parameter), and after observing data $\mathbf{x}$, we will have a new/updated belief distribution $\pi_{\Theta}(\theta \mid \mathbf{x})$. Let's see a picture of what MAP is going to do first, before getting more into the math and formalism.
### MLE vs MAP

(i) Which do you think is “better”, MLE or MAP? 
• Answer: There is no right answer. There are two main schools in statistics: Bayesians and Frequentists.
• Frequentists prefer MLE since they don’t believe you should be putting a prior belief on anything, and you should only make judgment based on what you’ve seen. They believe the parameter being estimated is a fixed quantity. 
• On the other hand, Bayesians prefer MAP, since they can incorporate their prior knowledge into the estimation. Hence the parameter being estimated is a random variable, and we seek the mode - the value with the highest probability or density. An example would be estimating the probability of heads of a coin - is it reasonable to assume it is more likely fair than not? If so, what distribution should we put on the parameter space?
• Anyway, in the long run, the prior “washes out”, and the only thing that matters is the likelihood; the observed data. For small sample sizes like this, the prior significantly influences the MAP estimate. However, as the number of samples goes to infinity, the MAP and MLE are equal.

Summary
- $MLE$: Chooses parameter that makes the observed data the most likely. Maximize the Likelihood
	- $\hat{\theta}_{M L E}=\arg \max _\theta p(D \mid \theta)$
	- Minimize the loss
- $MAP$: Maximize the Posterior probability
	- $\hat{\theta}_{M A P}=\arg \max _\theta p(D \mid \theta) p(\theta)$
	- Minimize the loss + regularilization (essentially Ridge Regression)


##### Marginal Liklihood

Marginal Liklihood is the evidence for a model $\mathcal{M}$

$$p(\mathcal{D} \mid \mathcal{M})=\int p(\boldsymbol{\theta} \mid \mathcal{M}) p(\mathcal{D} \mid \boldsymbol{\theta}, \mathcal{M}) d \boldsymbol{\theta}$$
##### Mixture of Conjugate Priors

Say we don't know if a coin was biased or not so we want the conjugate prior to be a mixture of differnet models . We could for example represent this as follows:

$$
p(\theta)=0.5 \operatorname{Beta}(\theta \mid 20,20)+0.5 \operatorname{Beta}(\theta \mid 30,10)
$$


If $\theta$ comes from the first distribution, the coin is fair, but if it comes from the second, it towards heads.

We can represent a mixture by introducing a latent indicator variable $h$, where $h=k$ me $\theta$ comes from mixture component $k$. The prior has the form

$$
p(\theta)=\sum_k p(h=k) p(\theta \mid h=k)
$$
#### Dirichlet Multinomial Model

Generalization from binary vairables to K-ary variables. Suppose $y$ it a categorical distrete random varialbe from a categorical distribution.

Liklihood:$$
p(\mathcal{D} \mid \boldsymbol{\theta})=\prod_{n=1}^N \operatorname{Cat}\left(y_n \mid \boldsymbol{\theta}\right)=\prod_{n=1}^N \prod_{c=1}^C \theta_c^{\mathbb{I}\left(y_n=c\right)}=\prod_{c=1}^C \theta_c^{N_c}
$$

where $N_c=\sum_n \mathbb{I}\left(y_n=c\right)$.

Prior: Dirichlet distribution which is a multivariate generlaization of the beta distribution:

$$
S_K=\left\{\boldsymbol{\theta}: 0 \leq \theta_k \leq 1, \sum_{k=1}^K \theta_k=1\right\}
$$


The pdf of the Dirichlet is defined as follows:

$$
\operatorname{Dir}(\boldsymbol{\theta} \mid \breve{\boldsymbol{\alpha}}) \triangleq \frac{1}{B(\breve{\boldsymbol{\alpha}})} \prod_{k=1}^K \theta_k^{\breve{\alpha}_k-1} \mathbb{I}\left(\boldsymbol{\theta} \in S_K\right)
$$

where $B(\breve{\boldsymbol{\alpha}})$ is the multivariate beta function,

$$
B(\breve{\boldsymbol{\alpha}}) \triangleq \frac{\prod_{k=1}^K \Gamma\left(\breve{\alpha}_k\right)}{\Gamma\left(\sum_{k=1}^K \breve{\alpha}_k\right)}
$$

Posterior: $$\begin{aligned}
&\text { We can combine the multinomial likelihood and Dirichlet prior to compute the posterior }\\
&\begin{aligned}
p(\boldsymbol{\theta} \mid \mathcal{D}) & \propto p(\mathcal{D} \mid \boldsymbol{\theta}) \operatorname{Dir}(\boldsymbol{\theta} \mid \breve{\boldsymbol{\alpha}}) \\
& =\left[\prod_k \theta_k^{N_k}\right]\left[\prod_k \theta_k^{\breve{a}_k-1}\right] \\
& =\operatorname{Dir}\left(\boldsymbol{\theta} \mid \breve{\alpha}_1+N_1, \ldots, \breve{\alpha}_K+N_K\right) \\
& =\operatorname{Dir}(\boldsymbol{\theta} \mid \overparen{\boldsymbol{\alpha}})
\end{aligned}
\end{aligned}$$

Refer to book for 
- Gaussian

### Beyond Simple Priors

Other kinds of priors where they aren't from same family or aren't easily interpretable

- Uninformative prior: $\text { such as a location parameter } \mu \in \mathbb{R} \text {, we can use a flat prior } p(\mu) \propto 1$
- Hierarchical priors: Bayesian models require specifying a prior $p(\boldsymbol{\theta})$ for the parameters. The parameters of the prior are called hyperparameters, and will be denoted by $\boldsymbol{\xi}$. If these are unknown, we can put a prior on them; this defines a hierarchical Bayesian model, or multi-level model, which can visualize like this: $\boldsymbol{\xi} \rightarrow \boldsymbol{\theta} \rightarrow \mathcal{D}$. We assume the prior on the hyper-parameters is fixed (e.g., we may use some kind of minimally informative prior), so the joint distribution has the form

$$
\begin{equation*}
p(\boldsymbol{\xi}, \boldsymbol{\theta}, \mathcal{D})=p(\boldsymbol{\xi}) p(\boldsymbol{\theta} \mid \boldsymbol{\xi}) p(\mathcal{D} \mid \boldsymbol{\theta}) 
\end{equation*}
$$


The hope is that we can learn the hyperparameters by treating the parameters themselves as datapoints. This is useful when we have multiple related parameters that need to be estimated (e.g., from different subpopulations, or muliple tasks); this provides a learning signal to the top level of the model.

- Empirical Priors: foundation of Empirical Bayes or Type II Maximum Liklihood. Middle ground between hierarchical and MLE. Estimate the hyperparameters of a prior distribution directly from the observed data rather than specifying them beforehand based on domain knowledge. 1. Hierarchical Structure: In a typical Bayesian model, we have parameters ( $\theta$ ) and hyperparameters $(\xi)$ arranged as $\xi \rightarrow \theta \rightarrow D_3$.
2. Point Estimation of Hyperparameters: Instead of treating hyperparameters as random variables and performing full Bayesian inference over them (which can be computationally expensive), we find a single "best" point estimate $\hat{\xi}$ by maximizing the marginal likelihood (also called the evidence) $1 \quad 2$.
3. The Optimization: The marginal likelihood is computed by integrating out the unknown parameters $\theta$ :

$$
\hat{\xi}=\operatorname{argmax}_{\xi} \int p(D \mid \theta) p(\theta \mid \xi) d \theta
$$

2.
4. Posterior Computation: Once $\hat{\xi}$ is estimated, we use it as a fixed value to compute the conditional posterior for the parameters: $p(\theta \mid \hat{\xi}, D) 1$.

Comparison of Methods
The sources categorize this approach in a hierarchy of "increasingly Bayesian" methods:
- Maximum Likelihood (MLE): Optimizes the likelihood $p(D \mid \theta)$ directly 4 .
- MAP Estimation: Optimizes $p(D \mid \theta) p(\theta \mid \xi)$ using fixed hyperparameters 4 .
- Empirical Bayes (ML-II): Optimizes the hyperparameters $\xi$ by maximizing the marginal likelihood 4.
- Full Bayes: Infers the joint posterior $p(\theta, \xi \mid D)$ by integrating over both parameters and hyperparameters 4 .

$$
\begin{array}{ll}
\text { Method } & \text { Definition } \\
\hline \text { Maximum likelihood } & \hat{\boldsymbol{\theta}}=\operatorname{argmax}_{\boldsymbol{\theta}} p(\mathcal{D} \mid \boldsymbol{\theta}) \\
\text { MAP estimation } & \hat{\boldsymbol{\theta}}(\boldsymbol{\xi})=\operatorname{argmax}_{\boldsymbol{\theta}} p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \boldsymbol{\xi}) \\
\text { ML-II (Empirical Bayes) } & \hat{\boldsymbol{\xi}}=\operatorname{argmax}_{\boldsymbol{\xi}} \int p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \boldsymbol{\xi}) d \boldsymbol{\theta} \\
\text { MAP-II } & \hat{\boldsymbol{\xi}}=\operatorname{argmax}_{\boldsymbol{\xi}} \int p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \boldsymbol{\xi}) p(\boldsymbol{\xi}) d \boldsymbol{\theta} \\
\text { Full Bayes } & p(\boldsymbol{\theta}, \boldsymbol{\xi} \mid \mathcal{D}) \propto p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \boldsymbol{\xi}) p(\boldsymbol{\xi})
\end{array}
$$

### Bayesian Machine Learning

In the sources, Bayesian machine learning is defined as the application of Bayesian principles to conditional models of the form $p(y \mid x, \theta) 1$. While standard machine learning typically focuses on finding a single "best" set of parameters, the Bayesian approach treats parameters ( $\theta$ ) as random variables and seeks to compute a probability distribution over them based on the observed data/

there is just a single best model, $\boldsymbol{\theta}$, such as the MLE. This is equivalent to approximating the posterior as an infinitely narrow, but infinitely tall, "spike" at the chosen value. We can write this as follows:

$$
\begin{equation*}
p(\boldsymbol{\theta} \mid \mathcal{D})=\delta(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}}) \tag{4.196}
\end{equation*}
$$

where $\delta$ is the Dirac delta function (see Section 2.6.5). If we use this approximation, then the predictive distribution can be obtained by simply "plugging in" the point estimate into the likelihood:

$$
\begin{equation*}
p(\boldsymbol{y} \mid \boldsymbol{x}, \mathcal{D})=\int p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta} \approx \int p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta}) \delta(\boldsymbol{\theta}-\hat{\boldsymbol{\theta}}) d \boldsymbol{\theta}=p(\boldsymbol{y} \mid \boldsymbol{x}, \hat{\boldsymbol{\theta}}) 
\end{equation*}
$$

##### Example: Binary Classification

Assume logistic Regression for binary classification. Assume Bernouli and Sigmoid Activation function. Then using MLE compute the decision boundary where The decision boundary is defined to be the input value $x^*$ where $p\left(y=1 \mid x^* ; \hat{\boldsymbol{\theta}}\right)=0.5$. We can solve for this value as follows:

$$
\begin{equation*}
\sigma\left(b+w x^*\right)=\frac{1}{1+e^{-\left(b+w x^*\right)}}=\frac{1}{2} 
\end{equation*}
$$
Problem
- We do not consider the uncertainity in the output probabilities and the location of the decision boundary. Approximate the Posterior $p(\theta | \mathcal{D})$ using a Monte Carlo Approximation where $\boldsymbol{\theta}^s \sim p(\boldsymbol{\theta} \mid \mathcal{D})$ is a posterior sample.

$p(y=1 \mid x, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S p\left(y=1 \mid x, \boldsymbol{\theta}^s\right)$



##### Approximating Posterior
- Grid Approximation
- Quadratic (Laplace) Approximation
- Variational Approximation
- Markov Chain Monte Carlo Approximation

##### Dirac Delta
where $\delta$ is the Dirac delta function, defined by

$$
\delta(x)= \begin{cases}+\infty & \text { if } x=0 \\ 0 & \text { if } x \neq 0\end{cases}
$$

where

$$
\int_{-\infty}^{\infty} \delta(x) d x=1
$$


A slight variant of this is to define

$$
\delta_y(x)= \begin{cases}+\infty & \text { if } x=y \\ 0 & \text { if } x \neq y\end{cases}
$$


Note that we have

$$
\delta_y(x)=\delta(x-y)
$$
