### Different Data Types
- Time Bars: Time bars oversample information during low-activity periods and undersample infor mation during highactivity periods. Second, time-sampled series often exhibit poor statistical properties, like serial correlation, heteroscedasticity, and non-normality of returns
- Tick Bars: Number of transactions taking place. Price changes over a fixed number of transactions may have a Gaussian distribution. Price changes over a fixed time period may follow a stable Paretian distribution, whose variance is infinite. Since the number of transactions in any time period is random, the above statements are not necessarily in disagreement
- Volume Bars: Total volumne being done. realized that sampling returns by volume achieved even better statistical properties (i.e., closer to an IID Gaussian distribution) than sampling by tick bars. Another reason to prefer volume bars over time bars or tick bars is that several market microstructure theories study the interaction between prices and volume.
- Dollar bars are formed by sampling an observation every time a pre-defined mar ket value is exchanged.
	- This will be more consistent over time
	- number of outstadning shares changes over time

So 
- time bars -> every x minutes.
- tick bars -> every x trades
- volume bares -> every x shares
These don't adapt to information flow

Tick Imbalance bars: Sampling date when order flow becomes unusually one sided
- Measures informed trading pressure and order flow toxicity

You're given ticks: $\left(p_t, v_t\right)$ (price, volume)
We define:

$$
b_t= \begin{cases}b_{t-1} & \text { if } \Delta p_t=0 \\ \frac{\left|\Delta p_t\right|}{\Delta p_t} & \text { if } \Delta p_t \neq 0\end{cases}
$$
- price changes and goes up then 1
- price changes and goes down then -1
- same assume same direction

Imbalance
	- $\theta_T=\sum_{t=1}^T b_t$
	- mostly buys, then large positive
	- mostly sells, then large negative
	- $\text { Expected imbalance }=(\text { expected \#ticks }) \times \text { (average trade direction) }$

Then we form a bar when
- $T^*=\min _T\left\{\left|\theta_T\right| \geq \mathbb{E}_0[T] \cdot\left|2 P\left(b_t=1\right)-1\right|\right\}$
	- Observed Imbalance >= Expected Imbalance
	- Suppose:
- Expected ticks per bar = 100
- Avg buy probability $=0.5$

Then: $\text { Expected imbalance }=100 \cdot(2 \cdot 0.5-1)=0$

Same concept applied to Volume/Dollar Imbalance Bars!!!

#### Tick Run Bars
- Montiory Sequence of buys in the overal volume and take samples when that sequence diverges from our expectation!
	- This is for when large traders will sweep the order book!

First, we define the length of the current run as

$$
\theta_T=\max \left\{\sum_{t \mid b_t=1}^T b_t,-\sum_{t \mid b_t=-1}^T b_t\right\}
$$


Second, we compute the expected value of $\theta_T$ at the beginning of the bar

$$
\mathrm{E}_0\left[\theta_T\right]=\mathrm{E}_0[T] \max \left\{\mathrm{P}\left[b_t=1\right], 1-\mathrm{P}\left[b_t=1\right]\right\}
$$


In practice, we can estimate $\mathrm{E}_0[T]$ as an exponentially weighted moving average of $T$ values from prior bars, and $\mathrm{P}\left[b_t=1\right]$ as an exponentially weighted moving average of the proportion of buy ticks from prior bars.

Third, we define a tick runs bar (TRB) as a $T^*$-contiguous subset of ticks such that the following condition is met:

$$
T^*=\underset{T}{\arg \min }\left\{\theta_T \geq \mathrm{E}_0[T] \max \left\{\mathrm{P}\left[b_t=1\right], 1-\mathrm{P}\left[b_t=1\right]\right\}\right\}
$$

where the expected count of ticks from runs is implied by $\max \left\{\mathrm{P}\left[b_t=1\right]\right.$, $\left.1-\mathrm{P}\left[b_t=-1\right]\right\}$. When $\theta_T$ exhibits more runs than expected, a low $T$ will satisfy these conditions. Note that in this definition of runs we allow for sequence breaks. That is, instead of measuring the length of the longest sequence, we count the number of ticks of each side, without offsetting them (no imbalance). In the context of forming bars, this turns out to be a more useful definition than measuring sequence lengths.

Same idea of Volume Run Bars and Dollar Run Bars!!!


### Sampling Features

Issues so far
- ML algos don't scale well with sample size
- ML algo achieve highest accuracy when learning from relevant examples


Reducing Data used to fit algo
- Sequeutning sampling
- Uniform sampling
- Event based sampling
	- CUSUM Filter: One practical aspect that makes CUSUM filters appealing is that multiple events are not triggered by gRaw hovering around a thresh old level, which is a flaw suffered by popular market signals such as Bollinger bands.

## Labelling Financial Data

Issues with time bars
- no good statistical properties
- same threshold $\tau$ is applied regardless of teh observed Volatility
- In other words, it is a very common error to label observations according to a f ixed threshold on time bars.

Better Alternatives
- label threshold using $\sigma_{t_i,0}$ estimate using rolling exponentially wieghted standard deviation of returns
- Use volume or dollars bars as volatility are closer to being homosckedastic
	- These both miss the path followed by prices
	- Need to set profit and stop losses 

Need to use a tripple barrier method
- two horizontal and one vertical
	- profit, loss, and time (number of bars per say)
	- 0 means barrier is not active!
	- Ex: [0,1,1] means exit for loss after certain number of bars or stopped out!


Learning Side and Size

Need ML to learn both the side and size of a bet! Snippet 3.3 implements the function getEvents, which finds the time of the first barrier touch.
- Pay attentiont to mpPandasObj function: multiprocessing function!

Suppose you know the side but need to to know size of the bet!
- Need to label this appropriately!!!
- only a binary prediciton!: Take a bet or no! 0,1!

Idea behind Meta Labelling: 2 spearate parts

- Primary Model: Determine the side of the position : buy or sell
- Secondary Model (Meta Labelling) : Determine whether to act on the position or pass on it!

ROC curve: measure cost of increasing true positive rate in terms of accepting the higher false positive rates. 
- The F1-score measures the efficiency of a classifier as the harmonic average between precision and recall
	- First build model that acieves high recall
	- then correct for low precission by applying meta labeling to the positives predicted by the primary model
		- Meta-labeling will increase your F1-score by filtering out the false positives, where the majority of positives have already been identified by the primary model. Stated differently, the role of the secondary ML algorithm is to determine whether a positive from the primary (exogenous) model is true or false.
		- Goal Determine whether to act or pass!

Benefits of Meta Labelling
- ML is beneficical to make MLs less black boxed
- Effects of overfitting are limited when you make ML metalabelling
	- For instance, consider that the features driving a rally may differ from the features driving a sell-off. In that case, you may want to develop an ML strategy exclusively for long positions, based on the buy recommendations of a primary model, and an ML strategy exclusively for short positions, based on the sell recommendations of an entirely different primary model.
- Achieving high accuracy on small bets and low accuracy on big bets will ruin you

In cases where certain labels are very rare
- prefarable to drop them for more common outcomes

## Sample Weights

Issue: financial data is not IID which is main assumption in majority of papers and ML algos. Whenever there is an overlap of two outcomes the series is not IID. 
- Spillage of information of one time period to another removes iid assumptions
- Time series has
	- Heteroscedasticity and non-normality
- Outcomes horizon must match frequency at which you sampled data

Solutions

- Find the number of concurrent labels
	- If two labels are functions of same price at a time $t$
		- For every time point $t$, a binary array is formed where an entry is 1 if a label overlaps with that time point else 0
		- $c_t$: Total count of labels that are overlapping at $t$
- Estimate the uniqueness of a label
	- Find average uniqueness of a label: reciprocal of the harmonic averic of $c_t$ over the event's lifespan i.e. 10 labels active means .1 uniqueness

Bagging classifiers

Suppose item $i$ from $I$ draws in a set of $I$ items:
- Probability of not selecting a particular $i$ : $(1-I^{-1})^{I}$
	- This converges to $e^{-1}$
- Number of unique observations is ${1-e^{-1}} = \frac{2}{3}$
- When you have max $K$ non-overalpping outcomes then : $1-e^{-\frac{K}{I}} \leq 1-e^{-1}$
	- **Implication is the incorrectly assuming IID draws leads to oversampling**
- Bootstrapping then means that
	- in bag observations are redundant to each other
	- very similar to out of bag observations
- Result
	- all trees in RF will be simple copies of a single overfit decision tree
	- Random Sampling makes out of bag similar to in bag so out of bag accuracy will be overinflated
- Solutions?
	- Drop overlapping outcomes before performing the bootstrap
		- Extreme loss of information. Don't do this
	- Ensure that frequency of sampling is equal to uniqueness. This reduces the undue influences of outcomes that contain redundant information. 
		- Set max_samples in sklearn.ensemble.Bagging Classfifier to equal the average uniqueness
	- Sequential Bootstrapping

Sequential Bootstrapping
- Draws are made according to changing probability that controls redundancy
	- Perform sequential sampling until $K$ distinct original observations appear. 
- Setup
	- First draw has equal probabilties
	- Update probabilties for the second draw, using potential uniqueness of every available item if it were added to the sequence of items already selected
	- Observations that have high overlap with the items already in the bag are assigned a lower porbability of being selected again
- Efficiency determined by using monte carlo experiments

Example:
- $y_1$: return of $0,3$
- $y_2$: return of 2,4
- $y_3$: return of 4,6

Overlaps: $\left\{1_{t, i}\right\}=\left[\begin{array}{lll}1 & 0 & 0 \\ 1 & 0 & 0 \\ 1 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 1\end{array}\right]$
Initially each has prob: 1/3

$$
1_{t, j}\left(1+\sum_{k \in \phi} 1_{t, k}\right)^{-1} 3 .
$$


For Label $y_1$ :
It overlaps with index 2 at $t=3$.
- $t=1: 1(1+0)^{-1}=1$
- $t=2: 1(1+0)^{-1}=1$
- $t=3: 1(1+1)^{-1}=\frac{1}{2}$
- Average Uniqueness: $\bar{u}_1^{(2)}=\frac{1+1+0.5}{3}=\frac{5}{6} \approx 0.833$ 4.

For Label $y_2$ :
Adding index 2 again results in perfect redundancy with itself at $t=3$ and $t=4$.
- $t=3: 1(1+1)^{-1}=\frac{1}{2}$
- $t=4: 1(1+1)^{-1}=\frac{1}{2}$
- Average Uniqueness: $\bar{u}_2^{(2)}=\frac{0.5+0.5}{2}=\frac{1}{2}=0.5$ 4.

Adding index 2 again results in perfect redundancy with itself at $t=3$ and $t=4$.
- $t=3: 1(1+1)^{-1}=\frac{1}{2}$
- $t=4: 1(1+1)^{-1}=\frac{1}{2}$
- Average Uniqueness: $\bar{u}_2^{(2)}=\frac{0.5+0.5}{2}=\frac{1}{2}=0.5 \quad 4$.

For Label $y_3$ :
It has zero overlap with index 2.
- $t=5: 1(1+0)^{-1}=1$
- $t=6: 1(1+0)^{-1}=1$
- Average Uniqueness: $\bar{u}_3^{(2)}=\frac{1+1}{2}=1$ 4.
4. Final Probability Calculation $\left(\delta^{(2)}\right)$

The probabilities are derived by scaling the potential uniqueness scores so they sum to 13 .
- Sum of potential uniqueness: $\frac{5}{6}+\frac{3}{6}+\frac{6}{6}=\frac{14}{6}$.
- $\delta_1^{(2)}=\frac{5 / 6}{14 / 6}=\frac{5}{14} \approx 0.357$
- $\delta_2^{(2)}=\frac{3 / 6}{14 / 6}=\frac{3}{14} \approx 0.214$
- $\delta_3^{(2)}=\frac{6 / 6}{14 / 6}=\frac{6}{14} \approx 0.428 \quad 4$.

Result: The algorithm assigns the lowest probability (21.4\%) to re-drawing the item already in the bag ( $y_2$ ) and the highest probability ( $42.8 \%$ ) to the label that is currently independent of the bag $\left(y_3\right)$.


#### Return atribution

Need to weight samples by a function of 
- uniqueness 
	- Highly overlapping should be given less weight to prevent them from having a diproportionate influence on the model
- absolute return
	- labels associated with large returns are more important for a strategy return then less weigth
The unscaled weight $\left(\tilde{w}_i\right)$ is:

$$
\tilde{w}_i=\sum_{t=t_{i, 0}}^{t_{i, 1}} \frac{r_{t-1, t}}{c_t}
$$


To ensure compatibility with standard ML libraries (like scikit-learn), these weights are then scaled so that their sum equals the total number of observations $I$ :

$$
w_i=\frac{\tilde{w}_i I}{\sum_{j=1}^I \tilde{w}_j}
$$


This scaling means that the average weight in the dataset remains 12 .

**his method does not work for "neutral" cases**
- If tried for netural cases then lower returns would be higher importance 

Steps
- Derive log returns
- sum return attributed to each label (return divided by concurrency)
- take absolute value of that sum
- re scale the final array to sum to number of observations

#### Time Decay

Sample weights should decay as new observations arrive. 
- Decay is constructed as a cumulative sum of uniqueness rather than simple chronological time
- decay is now measured by actual information added to the system
- Determined by a parameter $c$
	- $c=1$ no time decay
	- $c=0$ oldest observation has no influence


#### Class Weights

Apply a weight to underrepresented labels
- Classification problems where most important classes have the rarest occurences
- class_weight function in sklearn!


## Fractionally Differentiated Features

Goal: Preserve as much memory as possible while keeping stationary features

In signal processing
- We don't want to remove the memory!

Supervised learning algorithms typically require stationary features. The reason is that we need to map a previously unseen (unlabeled) observation to a collection of labeled examples, and infer from them the label of that new observation. If the features are not stationary, we cannot map the new observation to a large number of known examples.
- Stationary is a necessary not sufficient condition for the high performance of an ML algo
- Tradeoff between stationary and memory
	- differentiation makes it more stationary but we remove memory

Main Method: The method utilizes the backshift operator $B$, where $B^k X_t=X_{t-k}$ for any integer $k \geq 0$ 2 . In a standard integer model, $(1-B)^d$ would represent a simple $d$-order difference 2 . In the fractional model, $d$ is allowed to be a real number, using the formal binomial series expansion:

$$
(1-B)^d=\sum_{k=0}^{\infty}\binom{d}{k}(-B)^k=1-d B+\frac{d(d-1)}{2!} B^2-\frac{d(d-1)(d-2)}{3!} B^3+\ldots
$$

This series expansion results in a dot product of weights $(\omega)$ and historical values $(X)$ :
- Weights: $\left.\omega=\left\{1,-d, \frac{d(d-1)}{2!}, \ldots\right\}\right\}^3$
- Values: $X=\left\{X_t, X_{t-1}, X_{t-2}, \ldots\right\}$ 3

The sources contrast this with integer differentiation (e.g., $d=1$ for log-returns) $4 \quad 5$. In the integer case, the weights effectively "cut off" memory after a finite point (for $d=1$, all weights except $\omega_0=1$ and $\omega_1=-1$ are zero) 1 4. Fractional values of $d$ allow for a long memory because the weights do not drop to zero immediately, allowing distant historical levels to still influence the current differentiated value $\begin{array}{lll} & l^4 & \end{array}$.

Compute the weights iteratively as: $\omega_k=-\omega_{k-1} \frac{d-k+1}{k}$


In reality Implement this as follows:

1. Start with log prices rather than raw
	1. if using different type of data then use cumulative sum
2. Use Fixed Width Window Fracdiff
	1. $\tau$ = 10^{-5}. Drop any wieghts whose absolte value is below this threshold
	2. result of this is a non gaussian process with skew and kurtotis but is stationary
3. Conduct search to find optional $d$
	1.  Iterate throguh d from 0 to 1
	2. Test from stationary using ADF test
	3. Determinum minimum coefficient $d$ where p value of adf statistic falls below significance level of 5%

To summarize, most econometric analyses follow one of two paradigms:
1. Box-Jenkins: Returns are stationary, however memory-less.
2. Engle-Granger: Log-prices have memory, however they are non-stationary. Cointegration is the trick that makes regression work on non-stationary series, so that memory is preserved. However the number of cointegrated variables is limited, and the cointegrating vectors are notoriously unstable.

FFD
- Allows you to achieve stationary without renouncing too much memory
- Practice
	- cumulative sum of a time series
	- Compute FFD(d)
	- determine min $d$ such that $p-value$ of ADF statistic on $FFD(d)$ falls below 5%
	- use as predicitve feature


