Problem: Outliers in our data exist which are often labelling error introducing label noise. 

Solution: Robust Linear Regression
- Modify likelihood so that it predicts each output label $y$ is generated uniformly at random with probability $\pi$ and otherwise is generated using the usual conditional model:

Binary Logistic Regression:
$$
p(y \mid \boldsymbol{x})=\pi \operatorname{Ber}(y \mid 0.5)+(1-\pi) \operatorname{Ber}\left(y \mid \sigma\left(\boldsymbol{w}^{\top} \boldsymbol{x}\right)\right)
$$
- First Term means that it is pure noise and a coin flip
- Second term is that it is the usual Logistic Regression

Issues

1) Outliers that are far from boundary but are mislabeled contributed massively to the loss
- Solution:
	- Replace the usual cross entropy loss with a tempered version

Cross Entropy Loss: $$
\mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})=\mathbb{H}_{c e}(\boldsymbol{y}, \hat{\boldsymbol{y}})=\sum_c y_c \log \hat{y}_c
$$
Temepered Cross Entropy Loss: $$
\mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})=\sum_c\left[y_c\left(\log _{t_1} y_c-\log _{t_1} \hat{y}_c\right)-\frac{1}{2-t_1}\left(y_c^{2-t_1}-\hat{y}_c^{2-t_1}\right)\right]
$$
- $\log _t(x) \triangleq \frac{1}{1-t}\left(x^{1-t}-1\right)$
- If $y$ is once hot with all its mass on class $c$ then: $$
\mathcal{L}(c, \hat{\boldsymbol{y}})=-\log _{t_1} \hat{y}_c-\frac{1}{2-t_1}\left(1-\sum_{c^{\prime}=1}^C \hat{y}_{c^{\prime}}^{2-t_1}\right)
$$
- Increasing and concave:
	- t=1: standard log
- Bounded from below by $-1 / (1-t)$ for $- \leq t < 1$

2) Examples that are near decision boundary but mislabeled need to use a transfer function (that maps activation $R^C$ to probabilties $(0,1)$ that has heavier tails then softmax

Softmax: $\hat{y}_c=\frac{a_c}{\sum_{c^{\prime}=1}^C \exp \left(a_{c^{\prime}}\right)}=\exp \left[a_c-\log \sum_{c^{\prime}=1}^C \exp \left(a_{c^{\prime}}\right)\right]$
- $a$ is logits vector
- Tempered Softmax: $\hat{y}_c=\exp _{t_2}\left(a_c-\lambda_{t_2}(\boldsymbol{a})\right)$
	- Temperature parameter: $t_2 > 1> t_1$
	- $\exp _t(x) \triangleq[1+(1-t) x]_{+}^{1 /(1-t)}$
	- Use Bianry search to solve for $\lambda_{t_2}$: $\sum_{c=1}^C \exp _{t_2}\left(a_c-\lambda(\boldsymbol{a})\right)=1$

Combining bi-tempered softmax with tempered cross entropy is called bi-tempered logistic regression
