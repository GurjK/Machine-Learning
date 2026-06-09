
## Bayesian Decision Theory

$\mathcal{A}$: Possible set of actions
$\mathcal{H}$: State of nature
$l(h,a)$: loss function

Create a loss Matrix
- State
- Benefit
- Cost

For each possible action $a$ we compute posterior expected loss or risk as
$$
\begin{equation*}
\rho(a \mid \boldsymbol{x}) \triangleq \mathbb{E}_{p(h \mid \boldsymbol{x})}[\ell(h, a)]=\sum_{h \in \mathcal{H}} \ell(h, a) p(h \mid \boldsymbol{x}) 
\end{equation*}
$$


The optimal policy $\pi^*(\boldsymbol{x})$, also called the Bayes estimator or Bayes decision rule $\delta^*(\boldsymbol{x})$, specifies what action to take when presented with evidence $\boldsymbol{x}$ so as to minimize the risk:

$$
\begin{equation*}
\pi^*(\boldsymbol{x})=\underset{a \in \mathcal{A}}{\operatorname{argmin}} \mathbb{E}_{p(h \mid \boldsymbol{x})}[\ell(h, a)] 
\end{equation*}
$$


An alternative, but equivalent, way of stating this result is as follows. Let us define a utility function $U(h, a)$ to be the desirability of each possible action in each possible state. If we set $U(h, a)=-\ell(h, a)$, then the optimal policy is as follows:

$$
\begin{equation*}
\pi^*(\boldsymbol{x})=\underset{a \in A}{\operatorname{argmax}} \mathbb{E}_h[U(h, a)] 
\end{equation*}
$$
This is the maximum expected utility principle.

#### Classification Problems

Define the 0 or 1 loss where for each misclassification we get 1 and 0 otherwise. The posterior expected loss would be $$
\begin{aligned}
&\rho(\hat{y} \mid \boldsymbol{x})=p\left(\hat{y} \neq y^* \mid \boldsymbol{x}\right)=1-p\left(y^*=\hat{y} \mid \boldsymbol{x}\right)\\
&\text { Hence the action that minimizes the expected loss is to choose the most probable label: }\\
&\pi(\boldsymbol{x})=\underset{y \in \mathcal{Y}}{\operatorname{argmax}} p(y \mid \boldsymbol{x})
\end{aligned}
$$
This is the MAP estimate.More Generally, we get that $$
\ell_{00} p_0+\ell_{10} p_1<\ell_{01} p_0+\ell_{11} p_1
$$
Alternatively: Suppose we are risk average and we want to have a reject option then we can form the following loss function: $\ell\left(y^*, a\right)=\left\{\begin{array}{cc}0 & \text { if } y^*=a \text { and } a \in\{1, \ldots, C\} \\ \lambda_r & \text { if } a=0 \\ \lambda_e & \text { otherwise }\end{array}\right.$ where $\lambda_r$ is the cost of the reject action, and $\lambda_e$ is the cost of a classification error. Exercise 5.1 asks you to show that the optimal action is to pick the reject action if the most probable class has a probability below $\lambda^*=1-\frac{\lambda_r}{\lambda_c}$; otherwise you should just pick the most probable class. In other words, the optimal policy is as follows (known as Chow's rule [Cho70]):

$$
a^*= \begin{cases}y^* & \text { if } p^*>\lambda^*  \\ \text { reject } & \text { otherwise }\end{cases}
$$

where

$$
\begin{align*}
y^* & =\underset{y \in\{1, \ldots, C\}}{\operatorname{argmax}} p(y \mid x)  \tag{}\\
p^* & =p\left(y^* \mid x\right)=\max _{y \in\{1, \ldots, C\}} p(y \mid x)  \tag{}\\
\lambda^* & =1-\frac{\lambda_r}{\lambda_e} 
\end{align*}
$$

#### ROC Curves

$\tau$: determined by cost of a false positive and a false negative. 
- Boundary line for Classification. The Rule! Vertical Line.

|  |  | Estimate |  |
| :---: | :---: | :---: | :---: |
|  |  | 0 | 1 |
| Truth | 0 | TN/N=TNR=Spec | FP/N=FPR=Type I= Fallout |
|  | 1 | FN/P=FNR=Miss=Type II | TP/P=TPR=Sens=Recall |
- True Positive Rate ( Sensitivity): $T P R_\tau=p(\hat{y}=1 \mid y=1, \tau)=\frac{T P_\tau}{T P_\tau+F N_\tau}$
	- sensitivity reflects the probability than an event is correctly predicted _given that a sample is truly an event_.
	- Liklihood part of equation
- False Positive Rate (False Alarm Rate): $F P R_\tau=p(\hat{y}=1 \mid y=0, \tau)=\frac{F P_\tau}{F P_\tau+T N_\tau}$
- Precision =$\frac{TP}{TP + FP}$
- Positive Predictive Value: $P P V=\frac{\text { sensitivity } \text { × } \text { prevalence }}{(\text { sensitivity } \text { × } \text { prevalence })+((1-\text { specificity }) \times(1-\text { prevalence }))}$
	- - _sensitivity_: if the profile is truly STEM, what is the probability that it is correctly predicted?
	- _PPV_: if the profile was predicted as STEM, what is the probability that it is STEM?
	 

Plotting the TPR vs the FPR is called the ROC Curve.
- AUC: Area under curve. Maximum =1 is idea;
- EER: Valye where FPR = FNR, where diagonal line intersects ROC Curve


![[ROC Curve.png]]
- A and B are different models Like Logistic Regression and Random Forest
	- Ideally this area under curves is 0!

Class Imbalance
- set of negatives is usually much larger than set of positives
- ROC curve is unaffected by class imbalance
- In these cases it is more valuable to compute a Precession-Recall curve

##### Precision Recall Curves
- Precision: $\mathcal{P}(\tau) \triangleq p(y=1 \mid \hat{y}=1, \tau)=\frac{T P_\tau}{T P_\tau+F P_\tau}$
	- Proportion of positive results that were correctly classified
- Recall: $\mathcal{R}(\tau) \triangleq p(\hat{y}=1 \mid y=1, \tau)=\frac{T P_\tau}{T P_\tau+F N_\tau}$
	- Fraction of positives we actually detected
- Summarizing PR Curves as scalars
	- For a fixed recall level: Precision at K Score
	- Interpolated Precission: mean AP over a set of different PR Curves
	- F Scores: $$
F_\beta \triangleq\left(1+\beta^2\right) \frac{\mathcal{P} \cdot \mathcal{R}}{\beta^2 \mathcal{P}+\mathcal{R}}=\frac{\left(1+\beta^2\right) T P}{\left(1+\beta^2\right) T P+\beta^2 F N+F P}
$$


If we set $\beta=1$, we get the harmonic mean of precision and recall:

$$
\begin{aligned}
\frac{1}{F_1} & =\frac{1}{2}\left(\frac{1}{\mathcal{P}}+\frac{1}{\mathcal{R}}\right) \\
F_1 & =\frac{2}{1 / \mathcal{R}+1 / \mathcal{P}}=2 \frac{\mathcal{P} \cdot \mathcal{R}}{\mathcal{P}+\mathcal{R}}=\frac{T P}{T P+\frac{1}{2}(F P+F N)}
\end{aligned}
$$
- Harmonic Mean: ore conservative and requires both precission and recall to be high
	- Alternate $\beta$ : if precission more import than use .5 else 2
	- F score is affected by imbalance

![[ROC Curve and Precission Recall.png]]
- a) First, an appropriate cutoff can be determined based on one’s expectations regarding the importance of either sensitivity or specificity. This cutoff can then be used to make the qualitative predictions. Secondly, and perhaps more importantly, it allows a model to be assessed without having to identify the best cutoff. Commonly, the area under the ROC curve (AUC) is used to evaluate models.
- b) A poor model would result in a precision-recall curve that is in the vicinity of the horizontal grey line that is at the value of the observed prevalence (0.18 here). The area under the curve is used to summarize model performance. The best possible value is 1.0, while the worst is the prevalence. The area under this curve is 0.603.

## Regression Loss Functions

Assume set of actions, $A=H$, is equal to states of nature. The resulting decision rules can be used to compute the optimal parameters for an estimator to return or the optimal action for a robot to take. Assume Real Scaled ValueD!

1. L2 Loss: $\ell_2(h, a)=(h-a)^2$
	1. Risk: $\rho(a \mid \boldsymbol{x})=\mathbb{E}\left[(h-a)^2 \mid \boldsymbol{x}\right]=\mathbb{E}\left[h^2 \mid \boldsymbol{x}\right]-2 a \mathbb{E}[h \mid \boldsymbol{x}]+a^2$
	2. Optimal Decission: Minimum Mean Squared Error: $\frac{\partial}{\partial a} \rho(a \mid \boldsymbol{x})=-2 \mathbb{E}[h \mid \boldsymbol{x}]+2 a=0 \Rightarrow \pi(\boldsymbol{x})=\mathbb{E}[h \mid \boldsymbol{x}]=\int h p(h \mid \boldsymbol{x}) d h$
	3. Notes
		1. Sensitive to outliers!
2. L1 Loss:: $\ell_1(h, a)=|h-a|$
3. Huber Loss:$r =h=a$
	1. L2 for errors smaller than $\delta$ and equivalent to L1 for larger errors
	2. $\ell_\delta(h, a)=\left\{\begin{array}{cc}r^2 / 2 & \text { if }|r| \leq \delta \\ \delta|r|-\delta^2 / 2 & \text { if }|r|>\delta\end{array}\right.$

### Probablistic Loss Functions

Assume: set of actions is to pick a porbability distribution over some value of interest. So probabilisitc predicition or probabilistic forecasting, rather than a specific value. 
- Assume state of nature is distribution: $h=p(Y \mid \boldsymbol{x})$,
- Action is another distribution: $a=q(Y \mid \boldsymbol{x})$
- pick $q$ to minimize : $\mathbb{E}[\ell(p, q)]$ for each given $\boldsymbol{x}$.

1. Kullback Leibler Divergence: $D_{\mathbb{K L}}(p \| q) \triangleq \sum_{y \in \mathcal{Y}} p(y) \log \frac{p(y)}{q(y)}$
	1. $\begin{aligned} D_{\mathbb{K L}}(p \| q) & =\sum_{y \in \mathcal{Y}} p(y) \log p(y)-\sum_{y \in \mathcal{Y}} p(y) \log q(y) \\ & =-\mathbb{H}(p)+\mathbb{H}_{c e}(p, q) \\ \mathbb{H}(p) & \triangleq-\sum_y p(y) \log p(y) \\ \mathbb{H}_{c e}(p, q) & \triangleq-\sum_y p(y) \log q(y)\end{aligned}$
		1. $H(p)$: Entropy: Measure of uncertainity or variance of $p$. Maximum if $p$ is uniform and 0 if $p$ is degenerate or deterministic delta function
		2. $H_{ce}$ cross Entropy: Measure expected number of bits we need to use to compress a dataset coming from $p$ if we design using our code from $q$
		3. $KL$ extra number of bits we need to use to compress the data due to using incorrect distribution $q$
			1. KL=0, correctly predicted all future events

We set to Minimize the $D_{KL}$ but since H(p) is constant wrt 1 it can be ignored therefore solution is just minimizing cross entropy!

$$
\begin{equation*}
q^*(Y)=\underset{q}{\operatorname{argmin}} \mathbb{H}_{c e}(p(Y), q(Y)) 
\end{equation*}
$$


Now consider the special case in which the true state of nature is a degenerate distribution, which uts all its mass on a single outcome, say $c$, i.e., $h=p(Y)=\mathbb{I}(Y=c)$. This is often called a one-hot" distribution, since it turns "on" the $c$ 'th element of the vector, and leaves the other lements "off", as shown in Figure 2.1. In this case, the cross entropy becomes

$$
\begin{equation*}
\mathbb{H}_{c e}(\delta(Y=c), q)=-\sum_{y \in \mathcal{Y}} \delta(y=c) \log q(y)=-\log q(c) 
\end{equation*}
$$


This is known as the log loss of the predictive distribution $q$ when given target label $c$.

A proper Scoring rule is when the distribution $q$ matches true distribution $p$ is that the loss function is minimized iff $p=q$ with equality then $\ell(p, p) \leq \ell(p, q)$

2. Brier Score
	1. the log term in KL loss is sensitive to errors for low probability events therefore a common alternative is to use the Brier score
	2. Here: True distribution is a set of $N$ delta functions, $\boldsymbol{p}_n\left(\mathbf{Y}_n\right)=\delta\left(\mathbf{Y}_n-\boldsymbol{y}_n\right)$
	3. $\operatorname{BS}(\boldsymbol{p}, \boldsymbol{q}) \triangleq \frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C\left(q_{n c}-p_{n c}\right)^2=\frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C\left(q_{n c}-y_{n c}\right)^2$

### Choosing the Right Model

consider when we need to choose the right model

$M_0$: Null hypothesis
$M_1$: Alternative Hypothesis
- Pick if $p\left(M_1 \mid \mathcal{D}\right)>p\left(M_0 \mid \mathcal{D}\right)$ or $p\left(M_1 \mid \mathcal{D}\right) / p\left(M_0 \mid \mathcal{D}\right)>1$.

$B_{1,0} \triangleq \frac{p\left(\mathcal{D} \mid M_1\right)}{p\left(\mathcal{D} \mid M_0\right)}$: Bayes Factor/ Liklihood Ratio
- >1 Prefer model 1 else model 0
- Use this Scale to determine Decision Ruke

|      Bayes factor $B F(1,0)$       |       Interpretation        |
| :--------------------------------: | :-------------------------: |
|        $B F<\frac{1}{100}$         | Decisive evidence for $M_0$ |
|         $B F<\frac{1}{10}$         |  Strong evidence for $M_0$  |
| $\frac{1}{1 \rho}<B F<\frac{1}{3}$ | Moderate evidence for $M_0$ |
|        $\frac{1}{3}<B F<1$         |   Weak evidence for $M_0$   |
|             $1<B F<3$              |   Weak evidence for $M_1$   |
|             $3<B F<10$             | Moderate evidence for $M_1$ |
|              $B F>10$              |  Strong evidence for $M_1$  |
|             $B F>100$              | Decisive evidence for $M_1$ |

##### Multiple Models

Pick the most probable model is we have a 0-1 loss:
$$
\hat{m}=\underset{m \in \mathcal{M}}{\operatorname{argmax}} p(m \mid \mathcal{D})
$$

where

$$
p(m \mid \mathcal{D})=\frac{p(\mathcal{D} \mid m) p(m)}{\sum_{m \in \mathcal{M}} p(\mathcal{D} \mid m) p(m)}
$$

is the posterior over models. If the prior over models is uniform, $p(m)=1 /|\mathcal{M}|$, then the MAP . model is given by

$$
\hat{m}=\underset{m \in \mathcal{M}}{\operatorname{argmax}} p(\mathcal{D} \mid m)
$$


The quantity $p(\mathcal{D} \mid m)$ is given by

$$
p(\mathcal{D} \mid m)=\int p(\mathcal{D} \mid \boldsymbol{\theta}, m) p(\boldsymbol{\theta} \mid m) d \boldsymbol{\theta}
$$
this is the marginal likelihood or evidence for model $m$
- liklihood of the data averaged over all possible parameter values weight by the prior p$(\theta | m)$
- Marginal Liklihood is closely related to frequentist cross validation!
	- An overly complex model will overfit early examples and will then predict remaining ones poorly also getting a low CV score!

Bayesian Occam's razor
- Choose simpler model.Bayesian marginal liklihood supports this!

###### Information Criteria

Marginal Liklihood: $p(\mathcal{D} \mid m)=\int p(\mathcal{D} \mid \boldsymbol{\theta}, m) p(\boldsymbol{\theta}) d \boldsymbol{\theta},$
- Difficult to compute since need to integrate over entire parameter space!
- Sensitive to prior choice!
- Therefore use Information Criteria

Information criteria: $\mathcal{L}(m)=-\log p(\mathcal{D} \mid \hat{\boldsymbol{\theta}}, m)+C(m)$
- C(m): complexity penalty!
- Common to scale NLL by -2 to get the deviance:  $\operatorname{deviance}(m)=-2 \log p(\mathcal{D} \mid \boldsymbol{\theta}, m)$

BIC (Bayesian Information Criteria): 
- Simple approximation to the log marginal liklelihood
	- make gaussian approximation to posterior
	- Lowest value is the best usually!
	- If estimated on a training set then it gives the estimate of the model performance on a new training set
- $\log p(\mathcal{D} \mid m) \approx \log p\left(\mathcal{D} \mid \hat{\boldsymbol{\theta}}_{\text {map }}\right)+\log p\left(\hat{\boldsymbol{\theta}}_{\text {map }}\right)-\frac{1}{2} \log |\mathbf{H}|$
	- $H$ is Hessian of the negative log joint
- BIC Score:$J_{\mathrm{BIC}}(m)=\log p(\mathcal{D} \mid m) \approx \log p(\mathcal{D} \mid \hat{\boldsymbol{\theta}}, m)-\frac{D_m}{2} \log N$
- BIC Loss: $\mathcal{L}_{\mathrm{BIC}}(m)=-2 \log p(\mathcal{D} \mid \hat{\boldsymbol{\theta}}, m)+D_m \log N$
	- -2 * log liklihood + number of parameter * log number of obserivations
	- Second Term grows with the number of parameter: complexity penality!
		- Filters out overly complex models!
		- Prefers simpliciity compared to AIC

AIC (Akaike Information Criteria):
- Penalizes complex models less heavily than BIC, since regularization term is independent of $N$
- $\mathcal{L}_{\mathrm{AIC}}(m)=-2 \log p(\mathcal{D} \mid \hat{\boldsymbol{\theta}}, m)+2 D_m$

MDL (Minimum description Length)
- suppose we want to choose a model so that the sender can send some data to the receiver using the fewest number of bits.
- $\mathcal{L}_{\mathrm{MDL}}(m)=-\log p(\mathcal{D} \mid \hat{\boldsymbol{\theta}}, m)+\frac{D_m}{2} \log N$

Problems with BIC, AIC, MLD?
- Hard to compute degrees of freedom of a model!
	- Causes issues since parameters highly correlated so mapping to Fischer matrix and Hessian may be singular or close to singular
	- If singular then use Widely Applicable Information Criteria
- WAIC
	- The WAIC replaces the plug-in approximation to the marginal $\log$ likelihood, $\ell(m)=\sum_n \log p\left(y_n \mid \hat{\boldsymbol{\theta}}, m\right)$ with the the expected log pointwise predictive density or ELPD, defined as ELPD $(m)= \sum_{n=1}^N \log p\left(y_n \mid \mathcal{D}, m\right)=\sum_{n=1}^N \log \mathbb{E}_{\boldsymbol{\theta} \mid \mathcal{D}, m}\left[p\left(y_n \mid \boldsymbol{\theta}, m\right)\right]$, which is usually approximated by Monte Carlo. In addition, the complexity term is defined by $C(m)=\sum_{n=1}^N \log \mathbb{V}_{\boldsymbol{\theta} \mid \mathcal{D}, m}\left[p\left(y_n \mid \boldsymbol{\theta}, m\right)\right]$, which again is usually approximated by Monte Carlo. (The intuition for this is as follows: if, for a given datapoint $y_n$, the different posterior samples $\boldsymbol{\theta}_s$ make very different predictions, then the model is uncertain, and likely too flexible. The complexity term essentially counts how often this occurs.) The WAIC loss we want to minimize is defined as $\mathcal{L}_{\text {WAIC }}(m)=-2 \mathrm{LPPD}(m)+2 C(m)$.

Bayesian Significance Testing
- Computoing marignal likelihoods is imposible
- Goal estimate effect size which leads to difference in measuring say null vs alternate 

T Test
- Thus we will work with the differences, $d_i=e_i^1-e_i^2$. We assume $d_i \sim \mathcal{N}\left(\Delta, \sigma^2\right)$. We are interested in $p(\Delta \mid \boldsymbol{d})$, where $\boldsymbol{d}=\left(d_1, \ldots, d_N\right)$.

If we use an uninformative prior for the unknown parameters $(\Delta, \sigma)$, one can show that the posterior marginal for the mean is given by a Student distribution:

$$
p(\Delta \mid \boldsymbol{d})=\mathcal{T}_{N-1}\left(\Delta \mid \mu, s^2 / N\right)
$$

where $\mu=\frac{1}{N} \sum_{i=1}^N d_i$ is the sample mean, and $s^2=\frac{1}{N-1} \sum_{i=1}^N\left(d_i-\mu\right)^2$ is an unbiased estimate of the variance. Hence we can easily compute $p(|\Delta|>\epsilon \mid \boldsymbol{d})$, with a ROPE of $\epsilon=0.01$ (say). 

$\chi^2$ Test
- $p(\Delta>\epsilon \mid \mathcal{D})=\int_\epsilon^{\infty} p(\Delta \mid \mathcal{D}) d \Delta$
	- $\Delta = \theta_1 - \theta_2$
	- $D$ is all the data

# Frequentist Decision Theory

Average over $x$ instead of conditioning on it and treat the unknown state of nature as fixed $\theta$. We define the frequentist risk of an estimator $\delta$ given an unknown state of nature $\boldsymbol{\theta}$ to be the $\exp$ loss when applying that estimator to data $\boldsymbol{x}$, where the expectation is over the data, sampled $p(\boldsymbol{x} \mid \boldsymbol{\theta})$ :

$$
R(\boldsymbol{\theta}, \delta) \triangleq \mathbb{E}_{p(\boldsymbol{x} \mid \boldsymbol{\theta})}[\ell(\boldsymbol{\theta}, \delta(\boldsymbol{x}))]
$$
-Goal is estimate mean of Gaussian: 5 estimators for sample mean under MSE are best summarized by. We see that in general, the best estimator depends on the value of $\theta^*$, which is unknown. If $\theta^*$ is very close to $\theta_0$, then $\delta_3$ (which just predicts $\left.\theta_0\right)$ is best. If $\theta^*$ is within some reasonable range around $\theta_0$, then the posterior mean, which combines the prior guess of $\theta_0$ with the actual data, is best. If $\theta^*$ is far from $\theta_0$, the MLE is best.

Different Types of Risk
- Bayes Risk
- Maximum risk

#### Empirical Risk Minimization

the risk of an estimator , called population risk is $R\left(f, p^*\right)=R(f) \triangleq \mathbb{E}_{p^*(\boldsymbol{x}) p^*(\boldsymbol{y} \mid \boldsymbol{x})}[\ell(\boldsymbol{y}, f(\boldsymbol{x})]$
But since we can't have the true population we use the emprirical population giving us empirical risk $$
R(f, \mathcal{D}) \triangleq \mathbb{E}_{p_{\mathcal{D}}(\boldsymbol{x}, \boldsymbol{y})}\left[\ell(\boldsymbol{y}, f(\boldsymbol{x})]=\frac{1}{N} \sum_{n=1}^N \ell\left(\boldsymbol{y}_n, f\left(\boldsymbol{x}_n\right)\right)\right.
$$


Note that $R(f, \mathcal{D})$ is a random variable, since it depends on the training set.
A natural way to choose the predictor is to use

$$
\hat{f}_{\mathrm{ERM}}=\underset{f \in \mathcal{H}}{\operatorname{argmin}} R(f, \mathcal{D})=\underset{f \in \mathcal{H}}{\operatorname{argmin}} \frac{1}{N} \sum_{n=1}^N \ell\left(\boldsymbol{y}_n, f\left(\boldsymbol{x}_n\right)\right)
$$

where we optimize over a specific hypothesis space $\mathcal{H}$ of functions. This is called empirical risk minimization (ERM).

Give a fixed training set $D$ then $$
\begin{equation*}
f_{\mathcal{D}}^*=\underset{f \in \mathcal{H}}{\operatorname{argmin}} R(f, \mathcal{D}) 
\end{equation*}
$$


This is a random function, since $\mathcal{D} \sim p^*$ is random.
By adding and substracting terms, we can show that the expected risk of our chosen predictor compared to the best possible predictor can be decomposed into two terms, as follows:

$$
\begin{equation*}
\mathbb{E}_{\mathcal{D} \sim p^*}\left[R\left(f_{\mathcal{D}}^*\right)-R\left(f^{* *}\right)\right]=\underbrace{R\left(f_{\mathcal{H}}^*\right)-R\left(f^{* *}\right)}_{\mathcal{E}_{\mathrm{app}}(\mathcal{H})}+\underbrace{\mathbb{E}_{\mathcal{D} \sim p^*}\left[R\left(f_{\mathcal{D}}^*\right)-R\left(f_{\mathcal{H}}^*\right)\right]}_{\mathcal{E}_{\mathrm{est}}(\mathcal{H}, N)} 
\end{equation*}
$$


The first term, $\mathcal{E}_{\text {app }}(\mathcal{H})$, is the approximation error, which measures how closely $f_{\mathcal{H}}^*$ can model the true optimal function $f^{* *}$. The second term, $\mathcal{E}_{\text {est }}(\mathcal{H}, N)$, is the estimation error, which measures the error in finding the best function in the class, due to having a finite training set of size $N$ to evaluate performance. 
- Approximation Error: how close we can model true function
	- Decrease by using more expressive family of function but this can lead to overfitting which increases estimation error
- Estimation Error: Best function in class due to having limitied data
- Generalization Gap: Quantifies Degree of overfitting: $\operatorname{GenGap}(f)=R(f)-R\left(f, \mathcal{D}_{\text {train }}\right) \approx R\left(f, \mathcal{D}_{\text {test }}\right)-R\left(f, \mathcal{D}_{\text {train }}\right)$

Goal is to balance approximation and estimation error:
- Overfitting: Add complexity term to objective
	- $R_\lambda(f, \mathcal{D})=R(f, \mathcal{D})+\lambda C(f)$
	- $R_\lambda(\boldsymbol{\theta}, \mathcal{D})=-\frac{1}{N} \sum_{n=1}^N \log p\left(\boldsymbol{y}_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}\right)-\lambda \log p(\boldsymbol{\theta})$
		- If using a log loss function -> MAP estimation
- Structural Risk: Estimate the hyperparameters to minimize for the lowest achievable empirical risk
	- $\hat{\lambda}=\underset{\lambda}{\operatorname{argmin}} \min _\theta R_\lambda(\boldsymbol{\theta}, \mathcal{D})$
	- Empirical risk underestimates the populating risk, resulting in overfitting when we choose $\lambda$
		- Optimism of the training error: Really need the true population risk
	- Two Ways to estimate
		- Cross Validation
			- In regular testing and training if training data is small we won't be able to make accurate estimate of future performance. Solution is to use Cross Validation: Split training into $K$ folds. Then train on K-1 and evaluate on K. Repeat! $R_\lambda^{\mathrm{cv}} \triangleq \frac{1}{K} \sum_{k=1}^K R_0\left(\hat{\boldsymbol{\theta}}_\lambda\left(\mathcal{D}_{-k}\right), \mathcal{D}_k\right)$
			- Combinate all data together and re-estimate the model parameters using $\hat{\boldsymbol{\theta}}=\operatorname{argmin}_{\boldsymbol{\theta}} R_{\tilde{i}}(\boldsymbol{\theta}, \mathcal{D})$
		- Statistical Learning Theory:
			- Create an upper bound on the generalization gap with a certain probability, then confident choosing a minimization of empirical risk will have low population risk. 
			- Theorem 5.4.1. For any data distribution $p^*$, and any dataset $\mathcal{D}$ of size $N$ drawn from $p^*$, the probability that the generalization error of a binary classifier will be more than $\epsilon$, in the worst case, is upper bounded as follows:$$\begin{equation*}P\left(\max _{h \in \mathcal{H}}|R(h)-R(h, \mathcal{D})|>\epsilon\right) \leq 2 \operatorname{dim}(\mathcal{H}) e^{-2 N \epsilon^2} \tag{5.93}\end{equation*}$$where $R(h, \mathcal{D})=\frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(f\left(\boldsymbol{x}_i\right) \neq y_i^*\right)$ is the empirical risk, and $R(h)=\mathbb{E}\left[\mathbb{I}\left(f(\boldsymbol{x}) \neq y^*\right)\right]$ is the population risk.

#### Frequentist Hypothesis Testing

Idea: Split sample space into two regions
- Accept $H_0$
- Reject $H_0$

Type I vs Type II errors and Neyman-Pearson lemma
- Type I error: False positive
	- Reject the null hypothesis when the null hypothesis is true
	- Error rate $\alpha$ is significance.
	- $\begin{aligned}\alpha\left(\mu_0\right) & =p(\text { type I error })=p\left(\text { reject } H_0 \mid H_0 \text { is true }\right) \\& =p\left(\bar{X}(\tilde{\mathcal{D}})>x^* \mid \tilde{\mathcal{D}} \sim H_0\right) \\& =p\left(\frac{\bar{X}-\mu_0}{\sigma / \sqrt{N}}>\frac{x^*-\mu_0}{\sigma / \sqrt{N}}\right)\end{aligned}$
	- Probability of making a type I error is $\alpha$

- Type II error: False Negative
	- Error rate $\beta$
	- Do not reject the null hypothessis when the alternative hypothesis is true
	- $\beta\left(\mu_1\right)=p(\text { type II error })=p\left(\text { accept } H_0 \mid H_1 \text { is true }\right)=p\left(\bar{X}(\tilde{\mathcal{D}})<x^* \mid \tilde{\mathcal{D}} \sim H_1\right)$
- Power : $1-\beta(\mu_1)$
	- probability that we reject $H_0$ given that $H_1$ is true
	- ability to recongize null hypothesis is wrong

t-test statistic
- A single number that summarizes the sample data used to conduct the test hypothesis


Standard Error
- How far sample statistics (mean) deviates from the actual population mean

Null Hypothesis
- Test is null hypothesis is true given the data
- **A statement of no change and is 0 assumed true until evidence presents otherwise*
- Compare observed value using a test statistic and compare its observed value to one that we would expect if the data came from the null hypothesis.
- Define p value: probability under null hypothesis ob observing a test statistic that is as large or larger than actually oberved data:$\operatorname{pval} \triangleq \operatorname{Pr}\left(\operatorname{test}(\overline{\mathcal{D}}) \geq \operatorname{test}(\mathcal{D}) \mid \overline{\mathcal{D}} \sim H_0\right)$
	- Reject if under some significance level 

p value
- Probability of observing a test statistic
	- smallest vakue of $\alpha$ for which we can reject the $H_0$
	- We can compute the p-value using pval $=1-\Phi(\operatorname{test}(\mathcal{D}))$, where $\Phi$ is the cdf of the sampling distribution of the test statistic. This is called a one-sided p-value. In some case it can be more appropriate to use a two-sided p-value of the form pval $=\operatorname{Pr}(\operatorname{test}(\tilde{\mathcal{D}}) \geq \operatorname{test}(\mathcal{D}) \mid \tilde{\mathcal{D}} \sim \left.H_0\right)+\operatorname{Pr}\left(\operatorname{test}(\tilde{\mathcal{D}}) \leq-\operatorname{test}(\mathcal{D}) \mid \tilde{\mathcal{D}} \sim H_0\right)$, where we have assumed test $(\mathcal{D}) \geq 0$. For example, suppose we use test $(\tilde{\mathcal{D}})=\left(\hat{\theta}(\tilde{\mathcal{D}})-\theta_0\right) / \hat{\operatorname{se}}(\tilde{\mathcal{D}})$, where $\theta_0$ is the value for $\theta^*$ given $H_0$, and $\hat{\theta}$ is the MLE
- p value is not the likelihood of the data under the null hypothesis , so small values are interprettted to mean the $H_0$ is unlikely and therefore $H_1$ is likely
- True probability of the null hypothesis then we should use Bayes Rule
- If the prior is uniform, so $p\left(H_0\right)=p\left(H_1\right)=0.5$, this can be rewritten in terms of the likel ratio $L R=p\left(\mathcal{D} \mid H_0\right) / p\left(\mathcal{D} \mid H_1\right)$ as follows:$$p\left(H_0 \mid \mathcal{D}\right)=\frac{L R}{L R+1}$$
- p values of 0.05 can lead to large posterior probabilties of $H_0$ as high as 30
- 


Alternative Hyptothesis
- **A statement that the researcher is trying to find evidence to support**


Test
- One tailed test: Test statistics fall into one specified tail of the sampling distribution
- Two sided test: Test statistics fall into either tail of the sampling distribution
- Likelihood Ratio Test
	- Create Hypothesis Class: Suppose we have a uniform prior, so $p(H=0)=p(H=1)=0.5$, and that we use $0-1$ loss. Then he optimal decision rule is to accept $H_0$ iff $\frac{p\left(\mathcal{D} \mid H_0\right)}{n\left(\mathcal{D} \mid H^2\right)}>1$. This is called the likelihood ratio test.
- Significance Test
	- Define $H_0$ and $H_1$
	- Identify test, $\alpha$, find critical value and test statistic
	- Construct acceptance and rejectance regions
	- Calculate test statistics
		- Calculate p value
	- Retain or reject the hypothesis
- Chi Square Test for indepedence
	- Checks whether two categorical variables or related or not
- T-Test
	- Looks at the difference between two groups
- ANOVA
	- Test signifiance of group differences between two or more groups
	- Only determines that there is a difference not which is different
- ANCOVA
	- Same as Anova, but adds control of one or more of the covariates that may influence the dependent variable
- Z Test
	- $\boldsymbol{Z}$-test. If $\sigma^2$ is known, we use the $Z$-test: calculate $z=\frac{\sqrt{3}\left(\bar{x}-\mu_0\right)}{\sigma}$ and rejects $H_0$ in favour of $H_A$ if $|z|>1.96$, at significance level $\alpha=5 \%$.
	- The rejection region for the $Z-Test$ is the exterior of a slab defined by two parallel planes
		- $\left|x_1+x_2+x_3-3 \mu_0\right|>1.96 \sqrt{3} \sigma$ with boundaries $x_1+x_2+x_3=3 \mu_0 \pm 1.96 \sqrt{3} \sigma$ (for 3 variables)
		- Key: We assume we know the standard deviation here !!!!![[Z Test.png|293]]
		- Sepearate between two planes is determined by exact standard deviation and sample standard deviation plays no role here!!!
- T Test
	-  Suppose $X_1, X_2, X_3 \sim^{\text {iid }} N\left(\mu, \sigma^2\right)$ with unknown $\mu$ and $\sigma^2$. We wish to test$$H_0: \mu=\mu_0 \quad \text { vs } \quad H_A: \mu \neq \mu_0$$given observed values $x_1, x_2, x_3$. The standard two-sided $t$-test calculates the $t$-statistic$$t:=\frac{\sqrt{3}\left(\bar{x}-\mu_0\right)}{s}, \quad s^2=\frac{1}{n-1} \sum_{i=1}^3\left(x_i-\bar{x}\right)^2$$and rejects $H_0$ in favour of $H_A$ if $|t|>4.30$, at significance level $\alpha=5 \%$.
	- When sigma is unknown , check whether hypothesis mean is true mean. So here we estimate it using sample standard deviation
	- Angle between Red and Blue here , so 1,1,1, vector and the residual vector. This means that rejection region is a pair of cones
		- Narrow cone, smaller signifance level $\alpha$
		- ![[T Test.png|326]]
		- ![[T Test 2.png]]
- F Test
	- Null 




Example of Hypothesis Test: 
Suppose $X_1, X_2, X_3 \sim^{\text {iid }} N\left(\mu, \sigma^2\right)$. We wish to test

$$
H_0: \mu=\mu_0 \quad \text { vs } \quad H_A: \mu \neq \mu_0
$$

given observed values $x_1, x_2, x_3$.
- $\mu_0$ is given hypothesized value say 0.




#### Regression Metrics

- $RMSE$: Get result in correct units
- $R^2$: Standard correlation between observed and predicted values
	- For linear model: proportion of total variability in the outcome that can be explained by the model
	- Measure of correlation not accuracy
		- Possible that predicted values and observed have a strong linear relationship but do not confom to the 45 degree of agteement
		- Model under predicts at one extreme and overpredicts on other
			- Tree models
		- shows optimitistic results when outcome has large variance
		- misleading if there are a handful of outcomes values that are far away from thje overall scatter of the observed and predicted variables
	- Ex: Model where rideshares are higher during workweek but low on weekend inflates $R^2$
	- $RMSE and R^2$ can disagree
- $CCC$: Concordance correlation coefficient
	- Provides measure of correlation relative to the line of agreenment and is defined as the product of usual correlation coefficient and a measure of bias from the line of agreement
		- Futrhter the data deviates from line of agreement the smaller the bias
		- CC is a penalized version of correlation coefficient
		- Penality will apply if data exhibits poor correlation betweent he observed and predictes values or if relationship between predicted and observed values is far from line of agreement
- Robust methods attempt to lessen the value of outliers
	- Rank ordering:Transform extreme values so they appear closer to rest of data
	- Rank correlation
	- MAD

### Classification Metrics

- Confusion Matrix
	- Accuracy is misleading when classes are imbalanced
	- Cohen's Kappa:
		- Alternative to Accuracy
		- Normalizes error rate to what would be expected by chance
		- 1 completely agree and -1 completely disgree
	- Mosaic Plots
		- Each cell of the table is represented as a rectangle whose area is proportional to the number of values in the cell.
- 2 Class problems: soft predictions that can be converted toa  definitive class by choosing the class with largest probability
	- Binmoial log liklihood: $\log \ell=\sum_{i=1}^n \sum_{j=1}^C y_{i j} \log \left(p_{i j}\right)$
	- Gini Critierion
	- Entropy:
		- Only log should be maximizied while entropy and Gini minized