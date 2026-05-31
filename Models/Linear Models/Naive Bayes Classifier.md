
Main Assumption: Generative approach to classification is where we assume the features are conditionally indepedent on the given class label.
- This works even if features are no indepdent because it has a few parameters $C$ classes and $D$ features

Class has conditional density of the form: $p(\boldsymbol{x} \mid y=c, \boldsymbol{\theta})=\prod_{d=1}^D p\left(x_d \mid y=c, \boldsymbol{\theta}_{d c}\right)$
- $\theta_{dc}$ are params for the class conditional density for class $c$ and feature $d$

Therefore posterior is : $p(y=c \mid \boldsymbol{x}, \boldsymbol{\theta})=\frac{p(y=c \mid \boldsymbol{\pi}) \prod_{d=1}^D p\left(x_d \mid y=c, \boldsymbol{\theta}_{d c}\right)}{\sum_{c^{\prime}} p\left(y=c^{\prime} \mid \boldsymbol{\pi}\right) \prod_{d=1}^D p\left(x_d \mid y=c^{\prime}, \boldsymbol{\theta}_{d c^{\prime}}\right)}$
- $\pi$ is prior probability of class $c$

Probability distribution depends on feature $x_d$
- Binary: Bernouli
	- $\hat{\theta}_{d c}=\frac{N_{d c}}{N_c}$ which is the empirical fraction of times that feature $d$ is on in examples of class $c$.
- Categorical: Categorical
	- $\hat{\theta}_{d c k}=\frac{N_{d c k}}{\sum_{k^{\prime}=1}^K N_{d c k^{\prime}}}=\frac{N_{d c k}}{N_c}$ which is the number of times feature d had value k in example class c
- Real Value: Univariate Gaussian
	- $\begin{aligned} & \hat{\mu}_{d c}=\frac{1}{N_c} \sum_{n: y_n=c} x_{n d} \\ & \hat{\sigma}_{d c}^2=\frac{1}{N_c} \sum_{n: y_n=c}\left(x_{n d}-\hat{\mu}_{d c}\right)^2\end{aligned}$
Posterior: $p(y=c \mid \boldsymbol{x}, \mathcal{D}) \propto p(y=c \mid \mathcal{D}) \prod_d p\left(x_d \mid y=c, \mathcal{D}\right)=\bar{\pi}_c \prod_d \prod_k \bar{\theta}_{d c k}^{\mathbb{I}\left(x_d=k\right)}$
- osterior mean: $\bar{\theta}_{d c k}=\frac{\widehat{\beta}_{d c k}}{\sum_{k^{\prime}=1}^K \widehat{\beta}_{d c k^{\prime}}}=\frac{\breve{\beta}_{d c k}+N_{d c k}}{\sum_{k^{\prime}=1}^K \breve{\beta}_{d c k^{\prime}}+N_{d c k^{\prime}}}$

**Class Posterior $p(y | x \theta)$ for a NBC model has same form as a multinomial logisitic regression**
- Naive Bayes we optimize joint likelihood: $\prod_n p\left(y_n, \boldsymbol{x}_n \mid \boldsymbol{\theta}\right)$,  while logistic regression we optimizate conditional likelihood: $\prod_n p\left(y_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}\right)$


####