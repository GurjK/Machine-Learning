Models where the mean of the output $E[y |x.w]$ is a linear function of the inputs $x$ in both cases.

GLM is conditional version of an exponential family distribution where the natural parameters are a linear function of the input. The model has the form: 

$$p\left(y_n \mid \boldsymbol{x}_n, \boldsymbol{w}, \sigma^2\right)=\exp \left[\frac{y_n \eta_n-A\left(\eta_n\right)}{\sigma^2}+\log h\left(y_n, \sigma^2\right)\right]$$
- where $\eta_n \triangleq \boldsymbol{w}^{\top} \boldsymbol{x}_n$ is the (input dependent) natural parameter, $A\left(\eta_n\right)$ is the log normalizer, $\mathcal{T}\left(y_n\right)=y_n$ is the sufficient statistic, and $\sigma^2$ is the dispersion term. 

$l$: link function 
$l^{-1}$: mean function

Together the mapping from the linear inputs to the mean of outputs is done using $$\mu_n=\ell^{-1}\left(\eta_n\right)$$
Then we can show that the mean and variance of the response variable are as follows: $$\begin{aligned}
& \mathbb{E}\left[y_n \mid \boldsymbol{x}_n, \boldsymbol{w}, \sigma^2\right]=A^{\prime}\left(\eta_n\right) \triangleq \ell^{-1}\left(\eta_n\right) \\
& \mathbb{V}\left[y_n \mid \boldsymbol{x}_n, \boldsymbol{w}, \sigma^2\right]=A^{\prime \prime}\left(\eta_n\right) \sigma^2
\end{aligned}$$
#### Linear Regression

Recall that linear regression has the form
$$
p\left(y_n \mid \boldsymbol{x}_n, \boldsymbol{w}, \sigma^2\right)=\frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left(-\frac{1}{2 \sigma^2}\left(y_n-\boldsymbol{w}^{\top} \boldsymbol{x}_n\right)^2\right)
$$
Hence
$$
\log p\left(y_n \mid \boldsymbol{x}_n, \boldsymbol{w}, \sigma^2\right)=-\frac{1}{2 \sigma^2}\left(y_n-\eta_n\right)^2-\frac{1}{2} \log \left(2 \pi \sigma^2\right)
$$
where $\eta_n=\boldsymbol{w}^{\top} \boldsymbol{x}_n$. We can write this in GLM form as follows:
$$
\log p\left(y_n \mid \boldsymbol{x}_n, \boldsymbol{w}, \sigma^2\right)=\frac{y_n \eta_n-\frac{\eta_n^2}{2}}{\sigma^2}-\frac{1}{2}\left(\frac{y_n^2}{\sigma^2}+\log \left(2 \pi \sigma^2\right)\right)
$$

We see that $A\left(\eta_n\right)=\eta_n^2 / 2$ and hence
$$
\begin{aligned}
& \mathbb{E}\left[y_n\right]=\eta_n=\boldsymbol{w}^{\top} \boldsymbol{x}_n \\
& \mathbb{V}\left[y_n\right]=\sigma^2
\end{aligned}
$$

#### Binomial Regression
If the response variable is the number of successes in $N_n$ trials, $y_n \in\left\{0, \ldots, N_n\right\}$, we can use binomial regression, which is defined by
$$
p\left(y_n \mid \boldsymbol{x}_n, N_n, \boldsymbol{w}\right)=\operatorname{Bin}\left(y_n \mid \sigma\left(\boldsymbol{w}^{\top} \boldsymbol{x}_n\right), N_n\right)
$$

We see that binary logistic regression is the special case when $N_n=1$.
The log pdf is given by
$$
\begin{aligned}
\log p\left(y_n \mid \boldsymbol{x}_n, N_n, \boldsymbol{w}\right) & =y_n \log \mu_n+\left(N_n-y_n\right) \log \left(1-\mu_n\right)+\log \binom{N_n}{y_n} \\
& =y_n \log \left(\frac{\mu_n}{1-\mu_n}\right)+N_n \log \left(1-\mu_n\right)+\log \binom{N_n}{y_n}
\end{aligned}
$$
where $\mu_n=\sigma\left(\eta_n\right)$. To rewrite this in GLM form, let us define
$$
\eta_n \triangleq \log \left[\frac{\mu_n}{\left(1-\mu_n\right)}\right]=\log \left[\frac{1}{1+e^{-\boldsymbol{w}^{\top} \boldsymbol{x}_n}} \frac{1+e^{-\boldsymbol{w}^{\top} \boldsymbol{x}_n}}{e^{-\boldsymbol{w}^{\top} \boldsymbol{x}_n}}\right]=\log \frac{1}{e^{-\boldsymbol{w}^{\top} \boldsymbol{x}_n}}=\boldsymbol{w}^{\top} \boldsymbol{x}_n
$$


Hence we can write binomial regression in GLM form as follows
$$
\log p\left(y_n \mid \boldsymbol{x}_n, N_n, \boldsymbol{w}\right)=y_n \eta_n-A\left(\eta_n\right)+h\left(y_n\right)
$$
where $h\left(y_n\right)=\log \binom{N_n}{y_n}$ and

$$
A\left(\eta_n\right)=-N_n \log \left(1-\mu_n\right)=N_n \log \left(1+e^{\eta_n}\right)
$$Hence

$$
\mathbb{E}\left[y_n\right]=\frac{d A}{d \eta_n}=\frac{N_n e^{\eta_n}}{1+e^{\eta_n}}=\frac{N_n}{1+e^{-\eta_n}}=N_n \mu_n
$$

and

$$
\mathbb{V}\left[y_n\right]=\frac{d^2 A}{d \eta_n^2}=N_n \mu_n\left(1-\mu_n\right)
$$
#### Poisson Regression 

If the response variable is an integer count, $y_n \in\{0,1, \ldots\}$, we can use Poisson regression, is defined by

$$
p\left(y_n \mid \boldsymbol{x}_n, \boldsymbol{w}\right)=\operatorname{Poi}\left(y_n \mid \exp \left(\boldsymbol{w}^{\top} \boldsymbol{x}_n\right)\right)
$$

where

$$
\operatorname{Poi}(y \mid \mu)=e^{-\mu} \frac{\mu^y}{y!}
$$

is the Poisson distribution. Poisson regression is widely used in bio-statistical applications, $y_n$ might represent the number of diseases of a given person or place, or the number of read genomic location in a high-throughput sequencing context (see e.g., [Kua+09]).

The log pdf is given by
$$
\log p\left(y_n \mid \boldsymbol{x}_n, \boldsymbol{w}\right)=y_n \log \mu_n-\mu_n-\log \left(y_{n}!\right)
$$
where $\mu_n=\exp \left(\boldsymbol{w}^{\top} \boldsymbol{x}_n\right)$. Hence in GLM form we have
$$
\log p\left(y_n \mid \boldsymbol{x}_n, \boldsymbol{w}\right)=y_n \eta_n-A\left(\eta_n\right)+h\left(y_n\right)
$$
where $\eta_n=\log \left(\mu_n\right)=\boldsymbol{w}^{\top} \boldsymbol{x}_n, A\left(\eta_n\right)=\mu_n=e^{\eta_n}$, and $h\left(y_n\right)=-\log \left(y_{n}!\right)$. Hence
$$
\mathbb{E}\left[y_n\right]=\frac{d A}{d \eta_n}=e^{\eta_n}=\mu_n
$$
and

$$
\mathbb{V}\left[y_n\right]=\frac{d^2 A}{d \eta_n^2}=e^{\eta_n}=\mu_n
$$

- Common to use Deivance to measure performance
- We can also compute a calibration plot, which plots the actual frequency vs the predicted frequency.
## GLMs with non-canonical link functions

The canonical link function satisfies the property that $\theta = l(\mu)$ where $\theta$ are the canonical natural parameters. Therefore: $\theta=\ell(\mu)=\ell\left(\ell^{-1}(\eta)\right)=\eta$

Examples
- Bernouli
	- Canonical: log odds: $\theta=\log (\mu /(1-\mu))$
	- $\theta=\ell(\mu)=\operatorname{logit}(\mu)=\log \left(\frac{\mu}{1-\mu}\right)$
	- Inverse is the sigmoid function: $\mu=\sigma(\theta)=1 /\left(1+e^{-\theta}\right)$

Link functions
- Sigmoid: $\mu=\sigma(\theta)=1 /\left(1+e^{-\theta}\right)$
- Probit : $\eta=\ell(\mu)=\Phi^{-1}(\mu)$
- Complementary log logs odds: $\eta=\ell(\mu)=\log (-\log (1-\mu))$
	- Observe 0 events or one or more where events are assumed to governed by Poisson with $\lambda$

## MLE

GLMS are fit using Negative Log Likelihood estimation: 
$$
\mathrm{NLL}(\boldsymbol{w})=-\log p(\mathcal{D} \mid \boldsymbol{w})=-\frac{1}{\sigma^2} \sum_{n=1}^N \ell_n
$$
where
$$
\ell_n \triangleq \eta_n y_n-A\left(\eta_n\right)
$$

where $\eta_n=\boldsymbol{w}^{\top} \boldsymbol{x}_n$. For notational simplicity, we will assume $\sigma^2=1$.
We can compute the gradient for a single term as follows:

$$
\boldsymbol{g}_n \triangleq \frac{\partial \ell_n}{\partial \boldsymbol{w}}=\frac{\partial \ell_n}{\partial \eta_n} \frac{\partial \eta_n}{\partial \boldsymbol{w}}=\left(y_n-A^{\prime}\left(\eta_n\right)\right) \boldsymbol{x}_n=\left(y_n-\mu_n\right) \boldsymbol{x}_n
$$
- $\mu_n = f(w^Tx)$ and $f$ is the inverse link function mapping canonical parameters to mean parameters