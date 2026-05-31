Classification models of the form of a conditional density: $$
p(y=c \mid \boldsymbol{x}, \boldsymbol{\theta})=\frac{p(\boldsymbol{x} \mid y=c, \boldsymbol{\theta}) p(y=c \mid \boldsymbol{\theta})}{\sum_{c^{\prime}} p\left(\boldsymbol{x} \mid y=c^{\prime}, \boldsymbol{\theta}\right) p\left(y=c^{\prime} \mid \boldsymbol{\theta}\right)}
$$
- Called a generative classifier
	- way to generate features of $x$ for each class $c$ by sampling from $p(x| y,\theta)$
- $c$ is a class

- Discriminative classifier
	- directly models the class posterior $p( y| x. \theta)$

Resulting poster over classes if choosing conditional densities in a certain way is a linear function of $x$ :

$$
\log p(y=c \mid \boldsymbol{x}, \boldsymbol{\theta})=\boldsymbol{w}^{\top} \boldsymbol{x}+\mathrm{const},
$$
- $w$ comes from $\theta$

Called Linear Discriminant Analysis or LDA

### Gaussian Discriminant Analysis

Assume a generative classifier where the data comes from multivariate gaussian such that
- $p(\boldsymbol{x} \mid y=c, \boldsymbol{\theta})=\mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c\right)$

Resulting in a class posterior of $$
p(y=c \mid \boldsymbol{x}, \boldsymbol{\theta}) \propto \pi_c \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c\right)
$$
- $\pi$ is prior probability of label $c$

Now if we take the log of this we get that $$
\log p(y=c \mid \boldsymbol{x}, \boldsymbol{\theta})=\log \pi_c-\frac{1}{2} \log \left|2 \pi \boldsymbol{\Sigma}_c\right|-\frac{1}{2}\left(\boldsymbol{x}-\boldsymbol{\mu}_c\right)^{\top} \boldsymbol{\Sigma}_c^{-1}\left(\boldsymbol{x}-\boldsymbol{\mu}_c\right)+\mathrm{const}
$$
this is called the discriminant function. 

Insight
- The boundary between two classes will be a quadratic function of $x$
- In the rare cases where the covariance matrices are tied or shared across $c$ then $\Sigma$ is independent of $c$
	- Then we get that the decision boundary will be linear in $x$ leading to Linear Discriminant Analysis
	- $$
\begin{aligned}
\log p(y=c \mid \boldsymbol{x}, \boldsymbol{\theta}) & =\log \pi_c-\frac{1}{2}\left(\boldsymbol{x}-\boldsymbol{\mu}_c\right)^{\top} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{x}-\boldsymbol{\mu}_c\right)+\text { const } \\
& =\underbrace{\log \pi_c-\frac{1}{2} \boldsymbol{\mu}_c^{\top} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_c}_{\gamma_c}+\boldsymbol{x}^{\top} \underbrace{\boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_c}_{\boldsymbol{\beta}_c} \underbrace{+\operatorname{const}-\frac{1}{2} \boldsymbol{x}^{\top} \boldsymbol{\Sigma}^{-1} \boldsymbol{x}}_\kappa \\
& =\gamma_c+\boldsymbol{x}^{\top} \boldsymbol{\beta}_c+\kappa
\end{aligned}
$$
- Map Decision rule is $\hat{y}(x) = 1 iif w^tx > c$
- if $\pi = \pi_1$ then $c=\frac{1}{2} \boldsymbol{w}^{\top}\left(\boldsymbol{\mu}_1+\boldsymbol{\mu}_0\right)$
#### Connection between LDA and logistic regression

In LDA
- We first fit the Gaussian (class prior) to maximize the join likelihood, $p(x,y | \theta)$ then derivve $w$ from $\theta$
- learns the full probability distribution of the data and how features are distributed as gaussians and how common each class is.
	- After fitting, then the classifier weights are dettermined
- Correct is features are truly Gaussian
	- if not introduces bias
- Steps
	- Fit a gaussian per class
	- Estimate $\theta = \mu, \Sigma , \pi$
		- means, covariance, prior
	- Linear Classifier
In logistic Regression
- estimate $w$ directly to maximize the conditional likelihood $p(y | x,w)$
- asks given $x$ what is the probability of each class
- It estimated $w$ directly by maximizing the conditionally probability
- Steps
	- no model of $x$ as given
	- Estimate $w$ directly 
	- Linear classifier with no assumtipns

MLE
- $\hat{\boldsymbol{\mu}}_c=\frac{1}{N_c} \sum_{n: y_n=c} \boldsymbol{x}_n$
- $\hat{\boldsymbol{\Sigma}}_c=\frac{1}{N_c} \sum_{n: y_n=c}\left(\boldsymbol{x}_n-\hat{\boldsymbol{\mu}}_c\right)\left(\boldsymbol{x}_n-\hat{\boldsymbol{\mu}}_c\right)^{\top}$
	- Can overfit if $N_c$ small compared to $D$
	- If we force each of the classes to have the same covariances: then we get a more reliable estimation because we ar reducing the number of parameters that must be estiamteD:
		- $\hat{\boldsymbol{\Sigma}}=\frac{1}{N} \sum_{c=1}^C \sum_{n: y_n=c}\left(\boldsymbol{x}_n-\hat{\boldsymbol{\mu}}_c\right)\left(\boldsymbol{x}_n-\hat{\boldsymbol{\mu}}_c\right)^{\top}$
	- If we force $\Sigma_c$ to be diagonal then we reduce the number of parameters
		- Lose ability to capture correlations which is naive bayes assumption

MAP

Estimation of the full covariance Gaussian Matrix
- $\hat{\boldsymbol{\Sigma}}_{\text {map }}=\lambda \operatorname{diag}\left(\hat{\boldsymbol{\Sigma}}_{\text {mle }}\right)+(1-\lambda) \hat{\boldsymbol{\Sigma}}_{\text {mle }}$
- $\lambda$ is the regularization parameter
- this is known as RDA

Nearest Centroid Classifier
- Assume a uniform prior over class then can compute the most probable class as follows:
- $\hat{y}(\boldsymbol{x})=\underset{c}{\operatorname{argmax}} \log p(y=c \mid \boldsymbol{x}, \boldsymbol{\theta})=\underset{c}{\operatorname{argmin}}\left(\boldsymbol{x}-\boldsymbol{\mu}_c\right)^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}\left(\boldsymbol{x}-\boldsymbol{\mu}_c\right)$
	- Distance is measured with Mahalanobis distance and we are assigning $x$ to class with closest $\mu_c$
	- One simple Distance is to use: $d^2\left(\boldsymbol{x}, \boldsymbol{\mu}_c\right)=\left\|\boldsymbol{x}-\boldsymbol{\mu}_c\right\|_{\mathbf{W}}^2=\left(\boldsymbol{x}-\boldsymbol{\mu}_c\right)^{\mathrm{T}}\left(\mathbf{W} \mathbf{W}^{\mathrm{T}}\right)\left(\boldsymbol{x}-\boldsymbol{\mu}_c\right)=\left\|\mathbf{W}\left(\boldsymbol{x}-\boldsymbol{\mu}_c\right)\right\|^2$
		- Posterior: $p(y=c \mid \boldsymbol{x}, \boldsymbol{\mu}, \mathbf{W})=\frac{\exp \left(-\frac{1}{2}\left\|\mathbf{W}\left(\boldsymbol{x}-\boldsymbol{\mu}_c\right)\right\|_2^2\right)}{\sum_{c^{\prime}=1}^C \exp \left(-\frac{1}{2}\left\|\mathbf{W}\left(\boldsymbol{x}-\boldsymbol{\mu}_{c^{\prime}}\right)\right\|_2^2\right)}$
	- Then $W$ is learned with gradient descent and this is called nearest class mean metric learning:

Fishcer's LDA

Find matrix $W$ such that low dimensional data can be classified as well as possible using a gaussian class conditional density model
- Which line gives maximum separation between classes relative to the spread within classes?
- Gaussian is reasonalbe since we are compting linear combination of features
- 

Take features $x$ and reduce them to low dimensions features $z$ using a matrix $W$
- In 2d case this means finding a single vector $w$ onto which we can project the data
	- $\boldsymbol{w}=\mathbf{S}_W^{-1}\left(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1\right)$
- FLDA is restrited because it can find at most $K \leq C-1$ dimensional linear subspace no matter how large $D$

##### Generative vs Discriminative Classifiers

Generative:
- model: $p(\boldsymbol{x}, y)=p(y) p(\boldsymbol{x} \mid y)$
	- Can be used t generate samples $x$ from each class $y$
- Examples: 
	- Naive Bayes, LDA, QDA, Hidden Markov Models, Gaussian mixture models, Variations Encoders, GANs
- How the data was generated
	- $P(X |Y)$ and $P(Y)$ then $P(Y|X)$
	- Which class is most likely to produce $X$
- Pros
	- Easy to fit
		- Simpler mathematics
	- Can handle missing inputs easily
		- Marginalizing out the missing values
			- missingness of a feature is not informative about its potential value
	- Can fit classes separately
		- estimate parameters of each class independently
		- don't need to retrain if adding a class
	- Can handle unlabeled training data
		- 
	- May be robust to supurious feature
		- Can pick up on causal effects more


Discriminative
- model: $p(y \mid \boldsymbol{x})$
	- Can be used to discriminate between classes
- Learns boundary between classes
	- Given X which class
- Example:
	- Logistic, SVM, Neural Networks, Decision Trees, GBM
- Pros
	- Better Predictice Accuracy
		- Conditional distribution is often simpler then join
		- Don't need to model distribution of input features
	- Can handle feature preprocessing
		- Can perform polynomial transofrmations
	- Well callibrated probabilities estimates without prios
	

