
### Entropy

Information Content
- A dataset has high information content if the distribution from which it is drawn has high entropy

Entropy
- Entropy of a probability distribution is a measure of uncertainty or lack of predictability, associated with a random variable drawn from a given distribution
	- In reality, when a variable has many states it is very difficult to measure entropy


- For Discrete Random Variables $X$ with states $K$ from distribution $p$ , Entropy is 
$$\mathbb{H}(X) \triangleq-\sum_{k=1}^K p(X=k) \log _2 p(X=k)=-\mathbb{E}_X[\log p(X)]$$
- why log2?
	- The units are bits (short for binary digits)
- Maximum entropy Distribution?
	- Uniform because: $\mathbb{H}(X)=-\sum_{k=1}^K \frac{1}{K} \log (1 / K)=-\log (1 / K)=\log (K)$
	- Bernouli: Max entropy = 1
- Minimum entropy?
	- Any delta function which puts all mass into one state. No uncertainty in this distribution

Binary Entropy Function
- Consider a binary random variable $X \in (0,1)$ with $p(X=1) =\theta$ and $p(X=0) = 1-\theta$
- Then the Entropy of this is : $$\begin{aligned}
\mathbb{H}(X) & =-\left[p(X=1) \log _2 p(X=1)+p(X=0) \log _2 p(X=0)\right] \\
& =-\left[\theta \log _2 \theta+(1-\theta) \log _2(1-\theta)\right]
\end{aligned}$$

Cross Entropy
- Cross entropy between distribution $p$ and $q$ is $$\mathbb{H}_{c e}(p, q) \triangleq-\sum_{k=1}^K p_k \log q_k$$
- CE is the expected number of bits needed to compress some data samples drawn from distribution $p$ using a code based on distirbution $q$
	- Minimized when $p=q$

Join Entropy
 - $\begin{aligned}&\text { The joint entropy of two random variables } X \text { and } Y \text { is defined as }\\&\mathbb{H}(X, Y)=-\sum_{x, y} p(x, y) \log _2 p(x, y)\end{aligned}$

Conditional Entropy
- The conditional entropy of $Y$ given $X$ is the uncertainty we have in $Y$ after seeing $X$, averaged over possible values for $X$ :
	- $\mathbb{H}(Y \mid X) \triangleq \mathbb{E}_{p(X)}[\mathbb{H}(p(Y \mid X))] = =\mathbb{H}(X, Y)-\mathbb{H}(X)$
	- On average, conditioning on data never increases's one's uncertaintity

Perplexity
- The perplexity of a discrete probability distribution $p$ is defined as $\operatorname{perplexity}(p) \triangleq 2^{\mathbb{H}(p)}$
	- Measure of predictability. Lower bound is 1, if distribution can perfectly predict outcomes. 
	- Used to evaluate the quality of statistical language models, which is a generate model for sequence of tokens.
		- Long document $x$ of length $N$ and suppose $p$ is a simple unigram model then cross entropy is $H=-\frac{1}{N} \sum_{n=1}^N \log p\left(x_n\right)$ and perplexity is $\operatorname{perplexity}\left(p_{\mathcal{D}}, p\right)=2^H=2^{-\frac{1}{N} \log \left(\prod_{n=1}^N p\left(x_n\right)\right)}=\sqrt[N]{\prod_{n=1}^N \frac{1}{p\left(x_n\right)}}$
		- 

##### Differential Entropy for Continuous RV

$\begin{aligned}&\text { If } X \text { is a continuous random variable with pdf } p(x) \text {, we define the differential entropy }\\&h(X) \triangleq-\int_{\mathcal{X}} p(x) \log p(x) d x\end{aligned}$
- Can be negative here because pdfs can be greater than 1
- Ex: Uniform is $-\int_0^a 1/a log(1/a) = log(a)$
- Ex: Gaussian: $h\left(\mathcal{N}\left(\mu, \sigma^2\right)\right)=\frac{1}{2} \ln \left[2 \pi e \sigma^2\right]$
	- Increases as the variance but not always

##### Relative Entropy (KL divergence)

Two distributions $p$ and $q$ and goal is to measure how close they are. Let the metric to measure the divergence measure $D(p,q)$ which quantifies how far $p$ is from $q$.

##### Kullback Leiber divergence 

Known as information gain or relative entropy

For discrete distributions, the KL divergence is defined as follows:
$$
D_{\mathbb{K L}}(p \| q) \triangleq \sum_{k=1}^K p_k \log \frac{p_k}{q_k}
$$

This naturally extends to continuous distributions as well:

$$
D_{\mathbb{K L}}(p \| q) \triangleq \int d x p(x) \log \frac{p(x)}{q(x)}
$$
$$\begin{aligned}
&\text { We can rewrite the KL as follows: }\\
&D_{\mathbb{K} \mathbb{L}}(p \| q)=\underbrace{\sum_{k=1}^K p_k \log p_k}_{-\mathbb{H}(p)} \underbrace{-\sum_{k=1}^K p_k \log q_k}_{\mathrm{H}_{c e}(p, q)}
\end{aligned}$$
- First Term is negative Entropy
- Second Term is Cross Entropy
	- This is the lower bound on this. So interpret KL as extra number of bits you need to pay when compressing data samples if you use the incorrect distribution $q$ as the basis fo your coding scheme compared to true distribution $p$
- KL Divergence is always non-negative
- Gaussian:
	- Multivariate: $=\frac{1}{2}\left[\operatorname{tr}\left(\boldsymbol{\Sigma}_2^{-1} \boldsymbol{\Sigma}_1\right)+\left(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1\right)^{\top} \boldsymbol{\Sigma}_2^{-1}\left(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1\right)-D+\log \left(\frac{\operatorname{det}\left(\boldsymbol{\Sigma}_2\right)}{\operatorname{det}\left(\boldsymbol{\Sigma}_1\right)}\right)\right]$
	- Univariate: $=\log \frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+\left(\mu_1-\mu_2\right)^2}{2 \sigma_2^2}-\frac{1}{2}$
- Uniform Distribution maximizes the entropy

Task: Find distribution $q$ that is as close as possible to $p$, as measured by KL divergence:

$$
q^*=\arg \min _q D_{\mathrm{KL}}(p \| q)=\arg \min _q \int p(x) \log p(x) d x-\int p(x) \log q(x) d x
$$
- $p$ is empirical distribution that put probability on observed training and 0 everywhere else: $p_{\mathcal{D}}(x)=\frac{1}{N} \sum_{n=1}^N \delta\left(x-x_n\right)$

Then we get the **Cross Entropy Objective** :

$D_{\mathrm{KL}}\left(p_{\mathcal{D}} \| q\right)=-\frac{1}{N} \sum \log q\left(x_n\right)+C$
- $C=\int p(x) \log p(x) d x$ is a constant independent of $q$.
- Equal to the average negative log likelihood of $q$ on the training set

**Minimizing the KL Divergence to the empirical distribution is equivalent to maximizing likelihood**
- Flaws:
	- puts too much weight on the training set
	- we could smooth the empirical distribution using kernel density estimation but unlikely
	- Alternative Solution: **Data Augmentation**
		- perturbing the observed data samples in a way that we believe reflects plausible natural variation. Applying MLE on this augmented dataset yields superior results when fitting models with many parameters
		- See section 19.1 and 16.3 in probablistic machine learning!

Task: Approximate a distribution $p$ using simpler distribution $q$

We can accomplish this by minimizing $D_{\mathbb{K L}}(q \| p)$ or $D_{\mathbb{K L}}(p \| q)$.

Forward KL or Inclusive KL

$$
D_{\mathrm{KL}}(p \| q)=\int p(x) \log \frac{p(x)}{q(x)} d x
$$
- Minimizing wrt to $q$ is known as M projection
- This would cause q to cover p so we cant recover it 
- foreces $q$ to include all areas of space where $p$ is non zero or $q$ will be zero avoiding, overestimating support of $p$

Reverse KL or Exlcusive KL

$$
D_{\mathbb{K} \mathbb{L}}(q \| p)=\int q(x) \log \frac{q(x)}{p(x)} d x
$$
- $\text { Minimizing this wrt } q \text { is known as an I-projection or information projection. }$
	- Forces $q$ to exlcude all areas of space for which $p$ has zero prob
	- $q$ puts prob mass in very few parks of this space called zero forcing or mode seeking behavior

### Mutual Information

How to measure how dependent two random variables are?
- Look at the similarity of their distirbutions

$$
\begin{aligned}
&\text { The mutual information between rv's } X \text { and } Y \text { is defined as follows: }\\
&\mathbb{I}(X ; Y) \triangleq D_{\mathbb{K} \mathbb{L}}(p(x, y) \| p(x) p(y))=\sum_{y \in Y} \sum_{x \in X} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}
\end{aligned}
$$
Can also be repersented as

$$
\mathbb{I}(X ; Y)=\mathbb{H}(X)-\mathbb{H}(X \mid Y)=\mathbb{H}(Y)-\mathbb{H}(Y \mid X)
$$
or 
$$
\mathbb{I}(X ; Y)=\mathbb{H}(X)+\mathbb{H}(Y)-\mathbb{H}(X, Y)
$$

- Mutual Information is a KL divergence between joint and factored marginal distributions tells us that MI measures the information gain if we update from a model that treats two variables as independent $p(x)p(y)$ to one that uses their true join density $p(x,y)$

![[Entropy.png]]

Conditional Mutual Information
- $$
\begin{aligned}
\mathbb{I}(X ; Y \mid Z) & \triangleq \mathbb{E}_{p(Z)}[\mathbb{I}(X ; Y) \mid Z] \\
& =\mathbb{E}_{p(x, y, z)}\left[\log \frac{p(x, y \mid z)}{p(x \mid z) p(y \mid z)}\right] \\
& =\mathbb{H}(X \mid Z)+\mathbb{H}(Y \mid Z)-\mathbb{H}(X, Y \mid Z) \\
& =\mathbb{H}(X \mid Z)-\mathbb{H}(X \mid Y, Z)=\mathbb{H}(Y \mid Z)-\mathbb{H}(Y \mid X, Z) \\
& =\mathbb{H}(X, Z)+\mathbb{H}(Y, Z)-\mathbb{H}(Z)-\mathbb{H}(X, Y, Z) \\
& =\mathbb{I}(Y ; X, Z)-\mathbb{I}(Y ; Z)
\end{aligned}
$$
- $$
\mathbb{I}(Z, Y ; X)=\mathbb{I}(Z ; X)+\mathbb{I}(Y ; X \mid Z)
$$

- Conditional MI is the extra (residual) information that $X$ tells us about $Y$
- 1. $\rho=1$. In this case, $X=Y$, and $I(X, Y)=\infty$, which makes sense. Observing $Y$ tells us an infinite amount of information about $X$ (as we know its real value exactly).
2. $\rho=0$. In this case, $X$ and $Y$ are independent, and $I(X, Y)=0$, which makes sense. Observing $Y$ tells us nothing about $X$.
3. $\rho=-1$. In this case, $X=-Y$, and $I(X, Y)=\infty$, which again makes sense. Observing $Y$ allows us to predict $X$ to infinite precision.
	1. For scalar variables, a simple approximation is to discretize or quantize them, by dividing the ranges of each variable into bins, and computing how many values fall in each histogram bin . We can then easily compute the MI using the empirical pmf.
	2. alternatively, use K-nearest distances to estimate densities in non-parametric way. This is the **KSG estimator**

Normalized Mutual Information

Makes Mutual Information b/w 0 and 1
$$
N M I(X, Y)=\frac{\mathbb{I}(X ; Y)}{\min (\mathbb{H}(X), \mathbb{H}(Y))} \leq 1
$$
- When 0, $I(X,Y) =0$, so indepdent!
- NMI = 1, then $\mathbb{I}(X ; Y)=\mathbb{H}(X)-\mathbb{H}(X \mid Y)=\mathbb{H}(X) \Longrightarrow \mathbb{H}(X \mid Y)=0$
	- Only for Discrete RVS


Maximal Information Coefficient

Normalized estimate of Mutual Information
$$
\operatorname{MIC}(X, Y)=\max _G \frac{\mathbb{I}\left(\left.(X, Y)\right|_G\right)}{\log \|G\|}
$$
- where $G$ is the set of 2 d grids
- $\left.(X, Y)\right|_G$ represents a discretization of the variables onto this grid
- $\|G\|$ is $\min \left(G_x, G_y\right)$, where $G_x$ is the number of grid cells in the $x$ direction, and $G_y$ is the number of grid cells in the $y$ direction. 
- (The maximum grid resolution depends on the sample size $n$; they suggest restricting grids so that $G_x G_y \leq B(n)$, where $B(n)=n^\alpha$, where $\alpha=0.6$.) 
- The denominator is the entropy of a uniform joint distribution; dividing by this ensures $0 \leq$ MIC $\leq 1$.

If there is a relationship between $X$ and $Y$ there should be a discrete 2d input space that captures this
- MIC searches over grid sizes and boundaries
- MIC =0 means no relationship b/w variables
- MIC = 1 means noise free relationship of any form!

**Characteristic Matrix**: $M(k,l)$ is the maximum MI achievable by any grid size (k,l) normalized by log(min(k,l))
- MIC is max entropy in matrix
- Alternatively use MICe or TICe

Suppose we have an unknown variable $X$, and we observe a noisy function of it, call it $Y$. If we process the noisy observations in some way to create a new variable $Z$, it should be intuitively obvious that we cannot increase the amount of information we have about the unknown quantity, $X$. This is known as the data processing inequality. We now state this more formally, and then prove it.

Theorem 6.3.1. Suppose $X \rightarrow Y \rightarrow Z$ forms a Markov chain, so that $X \perp Z \mid Y$. Then $\mathbb{I}(X ; Y) \geq \mathbb{I}(X ; Z)$.
- Result of this is Sufficient Statistics:

Sufficient Statistics

An important consequence of the DPI is the following. Suppose we have the chain $\theta \rightarrow \mathcal{D} \rightarrow s(\mathcal{D})$. Then

$$
\begin{equation*}
\mathbb{I}(\theta ; s(\mathcal{D})) \leq \mathbb{I}(\theta ; \mathcal{D}) 
\end{equation*}
$$


If this holds with equality, then we say that $s(\mathcal{D})$ is a sufficient statistic of the data $\mathcal{D}$ for the purposes of inferring $\theta$. In this case, we can equivalently write $\theta \rightarrow s(\mathcal{D}) \rightarrow \mathcal{D}$, since we can reconstruct the data from knowing $s(\mathcal{D})$ just as accurately as from knowing $\theta$.
- minimal sufficient statistic: $s(\mathcal{D})$ contains no extra information about $\theta$
- thus $s(\mathcal{D})$ maximally compresses the data $\mathcal{D}$ without losing information which is relevant to predicting $\theta$. More formally, we say $s$ is a minimal sufficient statistic for $\mathcal{D}$ if for all sufficient statistics $s^{\prime}(\mathcal{D})$ there is some function $f$ such that $s(\mathcal{D})=f\left(s^{\prime}(\mathcal{D})\right)$. We can summarize the situation as follows:

$$
\begin{equation*}
\theta \rightarrow s(\mathcal{D}) \rightarrow s^{\prime}(\mathcal{D}) \rightarrow \mathcal{D} \tag{6.81}
\end{equation*}
$$
- EX: Bernouli Trials only need total number of flips
- Inferring mean of Gaussian with known variance only need empirical mean and number of samples