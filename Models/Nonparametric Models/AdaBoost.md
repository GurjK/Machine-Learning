Suppose interested in binary classification: predicting $\tilde{y} \in (-1,1)$. Weak learner computes
- $p(y=1 \mid \boldsymbol{x})=\frac{e^{F(\boldsymbol{x})}}{e^{-F(\boldsymbol{x})}+e^{F(\boldsymbol{x})}}=\frac{1}{1+e^{-2 F(\boldsymbol{x})}}$

so $F(x)$ returns half the log odds. The negative log liklihood is given by: $$
\ell(\tilde{y}, F(\boldsymbol{x}))=\log \left(1+e^{-2 \tilde{y} F(\boldsymbol{x})}\right)
$$
- We can minimize this by ensuring that margin: $m(x) = \tilde{y} F(x)$ is as large as possible
	- Positive margins are correctly classified
- Another loss could be the exponential loss: $$
\ell(\tilde{y}, F(\boldsymbol{x}))=\exp (-\tilde{y} F(\boldsymbol{x}))
$$
$$
\begin{aligned}
\frac{\partial}{\partial F(\boldsymbol{x})} \mathbb{E}\left[e^{-\tilde{y} f(\boldsymbol{x})} \mid \boldsymbol{x}\right] & =\frac{\partial}{\partial F(\boldsymbol{x})}\left[p(\tilde{y}=1 \mid \boldsymbol{x}) e^{-F(\boldsymbol{x})}+p(\tilde{y}=-1 \mid \boldsymbol{x}) e^{F(\boldsymbol{x})}\right] \\
& =-p(\tilde{y}=1 \mid \boldsymbol{x}) e^{-F(\boldsymbol{x})}+p(\tilde{y}=-1 \mid \boldsymbol{x}) e^{F(\boldsymbol{x})} \\
& =0 \Rightarrow \frac{p(\tilde{y}=1 \mid \boldsymbol{x})}{p(\tilde{y}=-1 \mid \boldsymbol{x})}=e^{2 F(\boldsymbol{x})}
\end{aligned}
$$
- Exponential Loss and Log Loss in the population setting have the same optimal solutions
	- Exponential loss is easier to optimize in boosting setting

How to solve for the $m$ th weak learner, $F_m$, when we use exponential loss
- Assume base classifier: binary class label ->  *Discrete AdaBoost*
- Assume base classifier: probability -> *Real AdaBoost*

At step $m$ we have to minimize: $$
L_m(F)=\sum_{i=1}^N \exp \left[-\tilde{y}_i\left(f_{m-1}\left(\boldsymbol{x}_i\right)+\beta F\left(\boldsymbol{x}_i\right)\right)\right]=\sum_{i=1}^N \omega_{i, m} \exp \left(-\beta \tilde{y}_i F\left(\boldsymbol{x}_i\right)\right)
$$
-  $\omega_{i, m} \triangleq \exp \left(-\tilde{y}_i f_{m-1}\left(\boldsymbol{x}_i\right)\right)$ is a weight applied to datacase $i$
	- Small weight means already classified well
	- Large weight means hard/misclassified
- $\tilde{y}_i \in\{-1,+1\}$
- In AdaBoost: $F\left(x_i\right) \in\{-1,+1\}$
	- $\tilde{y}_i F\left(x_i\right)= \begin{cases}+1 & \text { if correct classification } \\ -1 & \text { if misclassification }\end{cases}$
	- If Correct : +1 -> $exp (-\beta)$
		- Leads to a very small weight
	- if incorrect: -1 -> $exp(\beta)$
		- Leads to a larger weight


Then objective can be written as $$
\begin{aligned}
L_m & =e^{-\beta} \sum_{\tilde{y}_i=F\left(\boldsymbol{x}_i\right)} \omega_{i, m}+e^\beta \sum_{\tilde{y}_i \neq F\left(\boldsymbol{x}_i\right)} \omega_{i, m} \\
& =\left(e^\beta-e^{-\beta}\right) \sum_{i=1}^N \omega_{i, m} \mathbb{I}\left(\tilde{y}_i \neq F\left(\boldsymbol{x}_i\right)\right)+e^{-\beta} \sum_{i=1}^N \omega_{i, m}
\end{aligned}
$$
- First term is if we have correct points and second term is if we have inocrrect point
- Moving to second line
	- Factor out $e^{\beta}$
	- $\sum_i \omega_{i, m}=\sum_{\text {correct }} \omega_{i, m}+\sum_{\text {incorrect }} \omega_{i, m}$ -> $\sum_{\text {correct }} \omega=\sum_i \omega_i-\sum_{\text {incorrect }} \omega_i$
	- Start from:$$L_m=e^{-\beta} \sum_{\text {correct }} \omega_i+e^\beta \sum_{\text {incorrect }} \omega_i$$Replace correct sum:$$=e^{-\beta}\left(\sum_i \omega_i-\sum_{\text {incorrect }} \omega_i\right)+e^\beta \sum_{\text {incorrect }} \omega_i$$Distribute:$$=e^{-\beta} \sum_i \omega_i+\left(e^\beta-e^{-\beta}\right) \sum_{\text {incorrect }} \omega_i$$Now rewrite incorrect sum using indicator:$$\sum_{\text {incorrect }} \omega_i=\sum_i \omega_i \mathbb{I}\left(\tilde{y}_i \neq F\left(x_i\right)\right)$$So final form:$$L_m=\left(e^\beta-e^{-\beta}\right) \sum_{i=1}^N \omega_{i, m} \mathbb{I}\left(\tilde{y}_i \neq F\left(x_i\right)\right)+e^{-\beta} \sum_{i=1}^N \omega_{i, m}$$
	- The optimal Function $F$ to add here is: $F_m=\underset{F}{\operatorname{argmin}} \sum_{i=1}^N \omega_{i, m} \mathbb{I}\left(\tilde{y}_i \neq F\left(\boldsymbol{x}_i\right)\right)$
		- choose the next weak learner by minimizing weighted classification error
			- Among all weak learners, pick the one that best classifies highly weighted points
		- Indicator is 0 if classified correctly, and 1 if point is missclassified
			- $w_{im}$ if the point is wrong , 0 if correct
		- The whole sum is a weighted classification error
	- Normal classification minimizes: $\sum_i \mathbb{I}\left(y_i \neq F\left(x_i\right)\right)$
		- Total number of mistakes
	- AdaBoost minimizes: $\sum_i \omega_i \mathbb{I}\left(y_i \neq F\left(x_i\right)\right)$
		- Weighted mistakes
- Now solving for the step size, $\beta$ by substituing $F_m$ into $L_m$ we get: $\beta_m=\frac{1}{2} \log \frac{1-\operatorname{err}_m}{\operatorname{err}_m}$
	- $err_m <<.5$ then $\beta_m$ is large, so strong learners get more influence
	- If its close to .5 then $\beta_m \approx 0$
	- 
- $\operatorname{err}_m=\frac{\sum_{i=1}^N \omega_{i, m} \mathbb{I}\left(\tilde{y}_i \neq F_m\left(\boldsymbol{x}_i\right)\right)}{\sum_{i=1}^N \omega_{i, m}}$
	- fraction of weighted points misclassified
- Overall Update: $f_m(\boldsymbol{x})=f_{m-1}(\boldsymbol{x})+\beta_m F_m(\boldsymbol{x})$
- Recompute the wegihts for the next iteration: $\omega_{i, m+1}=e^{-\bar{y}_i f_m\left(\boldsymbol{x}_i\right)}=e^{-\bar{y}_i f_{m-1}\left(\boldsymbol{x}_i\right)-\bar{y}_i \beta_m F_m\left(\boldsymbol{x}_i\right)}=\omega_{i, m} e^{-\bar{y}_i \beta_m F_m\left(\boldsymbol{x}_i\right)}$
- Final Update: $\omega_{i, m+1}=\omega_{i, m} e^{\beta_m\left(2 \mathbb{I}\left(\bar{y}_i \neq F_m\left(\boldsymbol{x}_i\right)\right)-1\right)}=\omega_{i, m} e^{2 \beta_m \mathbb{I}\left(\bar{y}_i \neq F_m\left(\boldsymbol{x}_i\right)\right)} e^{-\beta_m}$
	- $\alpha_m=2 \beta_m$
	- Confidence we place on the predictive power of stump m
- Update is now: $\omega_{i, m+1}= \begin{cases}\omega_{i, m} e^{\alpha_m} & \text { if } \tilde{y}_i \neq F_m\left(\boldsymbol{x}_i\right) \\ \omega_{i, m} & \text { otherwise }\end{cases}$

*Exponentially increase weights of misclassified examples*

**Algorithm**

$$
\begin{aligned}
&\text { Algorithm 18.1: Adaboost.M1, for binary classification with exponential loss }\\
&\omega_i=1 / N\\
&\text { for } m=1: M \text { do }\\
&\text { Fit a classifier } F_m(\boldsymbol{x}) \text { to the training set using weights } \boldsymbol{w}\\
&\text { Compute } \operatorname{err}_m=\frac{\sum_{i=1}^N \omega_{i, m} \mathbb{1}\left(\tilde{y}_i \neq F_m\left(\boldsymbol{x}_i\right)\right)}{\sum_{i=1}^N \omega_{i, m}}\\
&\text { Compute } \alpha_m=\log \left[\left(1-\operatorname{err}_m\right) / \operatorname{err}_m\right]\\
&\text { Set } \omega_i \leftarrow \omega_i \exp \left[\alpha_m \mathbb{I}\left(\tilde{y}_i \neq F_m\left(\boldsymbol{x}_i\right)\right)\right]\\
&\text { Return } f(\boldsymbol{x})=\operatorname{sgn}\left[\sum_{m=1}^M \alpha_m F_m(\boldsymbol{x})\right]
\end{aligned}
$$
- Initial weights for each datapoint is $1/N$
	- Probability of selecting it during sampling
- Fit a weak learner: Usually just a decision tree stump
	- Update dataset weights
	- 


**Deriving Optimal $\beta$**

Rewrite loss:
$$
L_m=e^{-\beta}\left(1-\epsilon_m\right)+e^\beta \epsilon_m
$$

(assuming normalized weights)
Now minimize with respect to $\beta$.
Differentiate:
$$
\frac{d L}{d \beta}=-e^{-\beta}\left(1-\epsilon_m\right)+e^\beta \epsilon_m
$$
Set equal to zero:
$$
e^\beta \epsilon_m=e^{-\beta}\left(1-\epsilon_m\right)
$$
Multiply both sides by $e^\beta$ :
$$
e^{2 \beta} \epsilon_m=1-\epsilon_m
$$
Thus:
$$
e^{2 \beta}=\frac{1-\epsilon_m}{\epsilon_m}
$$
Take logs:
$$
2 \beta=\log \frac{1-\epsilon_m}{\epsilon_m}
$$
Final result:
$$
\beta_m=\frac{1}{2} \log \frac{1-\epsilon_m}{\epsilon_m}
$$


**Cons of AdaBoost**

Issue with exponential loss is that it puts a lot of weight on misclassified examples
- Very sensitive to outliers
- $e^{\tilde{y}f}$ is not the log of any pmf for binary variables $\tilde{y} \in (-1,+1)$
	- We cannot recover probability estimates from $f(x)$

Natural altenrative: use Log loss
- We can extract probabilities and makes mistakes linearly
- $$\begin{aligned}
&p(y=1 \mid \boldsymbol{x})=\frac{e^{f(\boldsymbol{x})}}{e^{-f(\boldsymbol{x})}+e^{f(\boldsymbol{x})}}=\frac{1}{1+e^{-2 f(\boldsymbol{x})}}\\
&\text { The goal is to minimze the expected log-loss, given by }\\
&L_m(F)=\sum_{i=1}^N \log \left[1+\exp \left(-2 \tilde{y}_i\left(f_{m-1}(\boldsymbol{x})+F\left(\boldsymbol{x}_i\right)\right)\right)\right]
\end{aligned}$$

ADABoost Algo Summary
- Initialize weights $w_i = 1 / N$ for every data point $i$
	- Start with null classfier: $f_0(\vec{x})=g_0(\vec{x})=0$
- For $t=0$ to $T-1$
	- Generate training dataset with $w_i$
	- Weights are lower/higher based on if they were classified wrong
	- Fit some weak learner $g_t$
- 


This is known as *logitBoost*

$$
\begin{aligned}
&\text { Algorithm 18.2: LogitBoost, for binary classification with log-loss }\\
&\omega_i=1 / N, \pi_i=1 / 2\\
&\text { for } m=1: M \text { do }\\
&\text { Compute the working response } z_i=\frac{y_i^*-\pi_i}{\pi_i\left(1-\pi_i\right)}\\
&\text { Compute the weights } \omega_i=\pi_i\left(1-\pi_i\right)\\
&F_m=\operatorname{argmin}_F \sum_{i=1}^N \omega_i\left(z_i-F\left(\boldsymbol{x}_i\right)\right)^2\\
&\text { Update } f(\boldsymbol{x}) \leftarrow f(\boldsymbol{x})+\frac{1}{2} F_m(\boldsymbol{x})\\
&\text { Compute } \pi_i=1 /\left(1+\exp \left(-2 f\left(\boldsymbol{x}_i\right)\right)\right) \text {; }\\
&\text { Return } f(\boldsymbol{x})=\operatorname{sgn}\left[\sum_{m=1}^M F_m(\boldsymbol{x})\right]
\end{aligned}
$$
