### Overview

For Feature Engineering a simple way to perform it is to replace $x$ with $\phi(x)$:
- Example: $\phi(x) = [1,x, x^2, x^3]$

Basis Function: $\phi(x)$

The model now becomes: $$f(x ;\theta) = W \phi(x) +b$$
- This is still linear parameters $\theta = (W,b)$
- This makes negative log likelihood which is convex

A natural extension is to endow the feature extractor with its parameters $\theta_2$
- $f(x; \theta) = W \phi (x ; \theta_2) +b$
where
- $\theta = ( \theta_1, \theta_2)$
- $\theta_1 = (W,b)$

We can repeat this recursively to get the following:
- $f(\boldsymbol{x} ; \boldsymbol{\theta})=f_L\left(f_{L-1}\left(\cdots\left(f_1(\boldsymbol{x})\right) \cdots\right)\right)$
by composing them L times.
- $f_{\ell}(\boldsymbol{x})=f\left(\boldsymbol{x} ; \boldsymbol{\theta}_{\ell}\right)$ is the function at layer $\ell$.

Deep Neural Networks (DNN)
- Compose differentiable functions into any kinf og DAG, mapping input to output
- Feedforward Neural Network or multilayer perceptron (MLP)
	- $x  \in R^D$
	- **Tabular Data**: Data is often stored $n \times D$ matrix
		- $n$: the number of samples
		- $D$: the number of features

Convolutional Neural Networks (CNN)
- Unstructured data such as images and text
Recurrning Neural Networks (RNN)
- Designed to work with sequences
Graph Neural Networks (GNN)
- Designed to work with graphs

### Multilayer perceptrons (MLPs)

Perceptron: mapping of the form
- $f(\boldsymbol{x} ; \boldsymbol{\theta})=\mathbb{I}\left(\boldsymbol{w}^{\top} \boldsymbol{x}+b \geq 0\right)=H\left(\boldsymbol{w}^{\top} \boldsymbol{x}+b\right)$
	- H(a): Heaviside step function or linear threshold function

Perceptrons can struggle to perform pattern recognition in certain cases like below:![[Perceptron.png]]

But to solve this we can stack multiple perceptrons on top of each other. 
- 3 perceptrons: h1, h2, y
- x: inputs
- 1: constants
	- one is -1.5 and the other is -.5
- h1 and h2: hidden units, values are not observed in hidden data
Layers
- $h_1=x_1 \wedge x_2$
	- approximately set weights
	- weights are initially 1
	- $\boldsymbol{w}_1^{\top} \boldsymbol{x}-b_1=[1.0,1.0]^{\top}[1,1]-1.5=0.5>0$
- $h_2=x_1 \vee x_2$, where $\vee$ is the OR operation
- $y=\overline{h_1} \wedge h_2$, where $\bar{h}=\neg h$ is the NOT (logical negation) operation.****

Therefore y computes: $y=f\left(x_1, x_2\right)=\overline{\left(x_1 \wedge x_2\right)} \wedge\left(x_1 \vee x_2\right)$
- This is the XOR function
- Therefore MLP can represent any logical function

What are issues with the Heavside?
- Not Differentiable! $H: R \in (0,1)$

##### Differentiable MLPs

One way to solve this issue it to replace the Heaviside with a differentiable activation function: $\varphi: \mathbb{R} \rightarrow \mathbb{R}$.
- Examples: Sigmoid, tanh, ReLU

Define hidden units $z_l$ at each layer $l$ to be a linear transformation of the hidden units at the previous layer passed elementwise through this activation function: $\boldsymbol{z}_l=f_l\left(\boldsymbol{z}_{l-1}\right)=\varphi_l\left(\boldsymbol{b}_l+\mathbf{W}_l \boldsymbol{z}_{l-1}\right)$

The quantity that is passed to the activation function is called the pre-activations: $\boldsymbol{a}_l=\boldsymbol{b}_l+\mathbf{W}_l \boldsymbol{z}_{l-1}$
Therefore,  $z_l=\varphi_l\left(a_l\right)$
- Compose L functions together
- **Backpropagation**: Compute gradient of output wrt to parameters in each layer: 
- Then pass gradient to optimizer and minimize some training objective

Steps
- Input to the layer: $z_l$
- Preactivation: $a _l = \left(\boldsymbol{b}_l+\mathbf{W}_l \boldsymbol{z}_{l-1}\right)$
	- Weighted sum of previous layer's outputs plus a bias term
- Activation Function: $\varphi(a_l)$
- Repeat this L Times so
	- $z_1=f_1\left(z_0\right), z_2=f_2\left(z_1\right), \ldots, z_1=f_1\left(z_1\right)$


##### Activation Functions

Key: Using a linear activation function: $\varphi_{\ell}(a)=c_{\ell} a$
- Then whole model is a regular linear model
Therefore we need to pick an Activation Function that is not Linear
- Can only add a straight hyperplane with a decision boundary with a linear activation
- **A neural network with at least one hidden layer and a non linear activation function can approximate any continuous function**
	- Each layer can specify a half plane and a sufficiently large combination of these can carve up any region of space
	- Reality: Deep Neural Networks work better than shallow ones
		- reason is that later layers can leverage the features that are learned by earlier layers; that is, the function is defined in a compositional or hierarchical way.
		- More Data the better the DNN

Activation functions
![[Activation Functions.png]]

Sigmoid: $\sigma(a)=\frac{1}{1+e^{-a}}$
- Ranges between 0 and 1
- Vanishes at 0 and 1. For large values this will always be 1

Tanh:
- Saturates at -1 and 1.

$\operatorname{ReLU}(a)=\max (a, 0)=a \mathbb{I}(a>0)$
- Turns off negative values
- Suffers from Spectral Bias
	- Fourier Transform leads to lower frequencies dominating compared to higher frequencies
	- To Fix this use a sin(wx): SIREN
		- Network has equal bandwidth at every frequency

Issues with saturation
- Gradients of output wrt input will be close to zero so any gradient signal from higher layers will not propagate back to earlier layers
	- Vanishing Gradient Problem
- Key: Use non-saturing activation functions
	- RELU

$$
\begin{array}{lll}
\text { Name } & \text { Definition } & \text { Range } \\
\hline \text { Sigmoid } & \sigma(a)=\frac{1}{1+e^{-a}} & {[0,1]} \\
\text { Hyperbolic tangent } & \tanh (a)=2 \sigma(2 a)-1 & {[-1,1]} \\
\text { Softplus } & \sigma_{+}(a)=\log \left(1+e^a\right) & {[0, \infty]} \\
\text { Rectified linear unit } & \operatorname{ReLU}(a)=\max (a, 0) & {[0, \infty]} \\
\text { Leaky ReLU } & \max (a, 0)+\alpha \min (a, 0) & {[-\infty, \infty]} \\
\text { Exponential linear unit } & \max (a, 0)+\min \left(\alpha\left(e^a-1\right), 0\right) & {[-\infty, \infty]} \\
\text { Swish } & a \sigma(a) & {[-\infty, \infty]} \\
\text { GELU } & a \Phi(a) & {[-\infty, \infty]}
\end{array}
$$


##### MLP for Classifying 2d data into 2 categories

Suppose we have 2 layers applied to a 2d vector

$$\begin{aligned} p(y \mid \boldsymbol{x} ; \boldsymbol{\theta}) & =\operatorname{Ber}\left(y \mid \sigma\left(a_3\right)\right) \\ a_3 & =\boldsymbol{w}_3^{\top} \boldsymbol{z}_2+b_3 \\ \boldsymbol{z}_2 & =\varphi\left(\mathbf{W}_2 \boldsymbol{z}_1+\boldsymbol{b}_2\right) \\ \boldsymbol{z}_1 & =\varphi\left(\mathbf{W}_1 \boldsymbol{x}+\boldsymbol{b}_1\right)\end{aligned}$$
- $a_3$ is the final logit score
	- Converted to a probability with sigmoid
- By adjusting the parameters: $\theta = W_1,b_1,W_2,b_2,w_3,b_3$ we can minimize the negative log likelihood, by fitting the training data very well, despite non-linear boundary

##### MLP for heteroskedastic regression

Heteroskedastic: Predicted output variance is input-dependent
- $f_\mu(\boldsymbol{x})=\mathbb{E}[y \mid \boldsymbol{x}, \boldsymbol{\theta}]$
- $f_\sigma(\boldsymbol{x})=\sqrt{\mathbb{V}[y \mid \boldsymbol{x}, \boldsymbol{\theta}]}$.

By using a common backbone, we can share the layers between these two parameters
- $\mu$ : $\varphi(a)=a$.
- $\sigma$: $\varphi(a)=\sigma_{+}(a)=\log \left(1+e^a\right)$

Overall Modell: $$
p(y \mid \boldsymbol{x}, \boldsymbol{\theta})=\mathcal{N}\left(y \mid \boldsymbol{w}_{\boldsymbol{\mu}}^{\top} f\left(\boldsymbol{x} ; \boldsymbol{w}_{\text {shared }}\right), \sigma_{+}\left(\boldsymbol{w}_{\boldsymbol{\sigma}}^{\top} f\left(\boldsymbol{x} ; \boldsymbol{w}_{\text {shared }}\right)\right)\right)
$$
- Good for seasonal oscilations and variance growing quadratically
- Ex: Stochastic Volatility Model

##### ANN

Artificial Neural Networks or ANNs and real neural networks are connected

Think of a neuron as a model called the **McCullocj-Pitts model**
- Neuron $k$ fires, denoted by $h_k \in (0,1)$ depending on inputs $x$ and strength of connections $w$
- Compute weighted sum $a_k = w_k^Tx$
	- Weights are wires from $x_d$ to $h_k$
	- Analogous to dendrites in a real neuron
- Wegihtes sum , $a_k$ comapred to a threshold $b_k$ and if activation exceeds threshold then neuron fires
	- Neuron emittting an electrical output or action potential
- Model behavior of a neuron as: $h_k(x) = H(w_k^T -b_k)$
	- $H(a) = I(a>0)$

How do ANNs differ
- Use Backpropagation, strictly feedforward


### Backpropagation

**Backpropagation**: Compute the gradient of the loss function applied to the output of the network wrt the parameters in each layer
- Gradient is passed to a gradient-based optimization algorithm
- Two Modes in Automatic Differnetiation
	- Forward Mode
	- Reverse Mode

**Forward vs Reverse Mode**
![[Linear Chain FeedForward Model.png]]
Consider a mapping of the form $o = f(x)$ where $x \in R^n$ and $o \in R^m$
- $\boldsymbol{f}=\boldsymbol{f}_4 \circ \boldsymbol{f}_3 \circ \boldsymbol{f}_2 \circ \boldsymbol{f}_1$

We need to compute $o=f_4(x),x_2 = f_1(x), x_3 = f_2(x),x_4 = f_3(x)$

Find the Jacobian: Find all first order partial derivatives of a vector value function with respect to its inputs and arrange them in a matrix:

$$
\begin{aligned}
\frac{\partial \boldsymbol{o}}{\partial \boldsymbol{x}} & =\frac{\partial \boldsymbol{o}}{\partial \boldsymbol{x}_4} \frac{\partial \boldsymbol{x}_4}{\partial \boldsymbol{x}_3} \frac{\partial \boldsymbol{x}_3}{\partial \boldsymbol{x}_2} \frac{\partial \boldsymbol{x}_2}{\partial \boldsymbol{x}}=\frac{\partial \boldsymbol{f}_4\left(\boldsymbol{x}_4\right)}{\partial \boldsymbol{x}_4} \frac{\partial \boldsymbol{f}_3\left(\boldsymbol{x}_3\right)}{\partial \boldsymbol{x}_3} \frac{\partial \boldsymbol{f}_2\left(\boldsymbol{x}_2\right)}{\partial \boldsymbol{x}_2} \frac{\partial \boldsymbol{f}_1(\boldsymbol{x})}{\partial \boldsymbol{x}} \\
& =\mathbf{J}_{\boldsymbol{f}_4}\left(\boldsymbol{x}_4\right) \mathbf{J}_{\boldsymbol{f}_3}\left(\boldsymbol{x}_3\right) \mathbf{J}_{\boldsymbol{f}_2}\left(\boldsymbol{x}_2\right) \mathbf{J}_{\boldsymbol{f}_1}(\boldsymbol{x})
\end{aligned}
$$
To compute the Jacobian $J_f(x)$ we need to remember that $$
\mathbf{J}_{\boldsymbol{f}}(\boldsymbol{x})=\frac{\partial \boldsymbol{f}(\boldsymbol{x})}{\partial \boldsymbol{x}}=\left(\begin{array}{ccc}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{array}\right)=\left(\begin{array}{c}
\nabla f_1(\boldsymbol{x})^{\top} \\
\vdots \\
\nabla f_m(\boldsymbol{x})^{\top}
\end{array}\right)=\left(\frac{\partial \boldsymbol{f}}{\partial x_1}, \cdots, \frac{\partial \boldsymbol{f}}{\partial x_n}\right) \in \mathbb{R}^{m \times n}
$$
- $\nabla f_i (x)$ is the ith row from $J_f(x)$
	- Derivatives of one output wrt all inputs
	- If I slightly move one input how does output, $f_i$ change 
- $\frac{\partial \boldsymbol{f}}{\partial x_j} \in \mathbb{R}^m$ is the $j$ 'th column (for $j=1: n$ )
	- Derivatives of all outputs wrt to one input
	- Of I move input $x_1$ how do all outputs react

Instead of computing the entire Jacobian matrix directly: $m \times n$
- We can recover this by 
	- computing columns one at a time or rows one at a time

**Jacobian Vector Product**

$J_f(x)v$
- Let $v$ represent the unit vector $v = (1, 0, 0)^T$
	- This grabs column 1
- Repeat this for each column then we get the entire Jacobian: n $JVPs$ = Entire Jacobian

$$
\mathbf{J}_f(x) v=\underbrace{\mathbf{J}_{f_4}\left(\boldsymbol{x}_4\right)}_{m \times m_3} \underbrace{\mathbf{J}_{f_3}\left(\boldsymbol{x}_3\right)}_{m_3 \times m_2} \underbrace{\mathbf{J}_{f_2}\left(\boldsymbol{x}_2\right)}_{m_2 \times m_1} \underbrace{\mathbf{J}_{f_1}\left(\boldsymbol{x}_1\right)}_{m_1 \times n} \underbrace{\boldsymbol{v}}_{n \times 1}
$$


**Vector Jacobian Product**

$u_T J_f(x)$
- Let v represent the unit vector $u = (1,0,0)$
	- This grabs first row
- $m$ VJPS -> Entire Jacobian

$$
\boldsymbol{u}^{\top} \mathbf{J}_{\boldsymbol{f}}(\boldsymbol{x})=\underbrace{\boldsymbol{u}^{\top}}_{1 \times m} \underbrace{\mathbf{J}_{\boldsymbol{f}_4}\left(\boldsymbol{x}_4\right)}_{m \times m_3} \underbrace{\mathbf{J}_{\boldsymbol{f}_3}\left(\boldsymbol{x}_3\right)}_{m_3 \times m_2} \underbrace{\mathbf{J}_{\boldsymbol{f}_2}\left(\boldsymbol{x}_2\right)}_{m_2 \times m_1} \underbrace{\mathbf{J}_{\boldsymbol{f}_1}\left(\boldsymbol{x}_1\right)}_{m_1 \times n}
$$
If $n<m$ (more outputs then inputs)
- USE JVP (forward mode)
If $n>m$ (more inputs then outputs)
- Use Backward Mode (VJP)
- 1 row is cheap compared to 100,000 columns for a Neural Network with 100,000 parameters
- Neural Networks use Backpropagation for this reason

*Neural Networks use VJP because computing 1 row is cheaper then 100,000 columns!*


During Forward Pass
- Information moves Down from Layer 1 to Layer 50
During BackProp
- Gradient moves upward
- At each layer gradient gets multiplied by Jacobian
- $g_l = J_l g_{l+1}$
	- Gradient at layer $l$ is gradient at layer $l+1$ * local derivative information


**Reverse Mode Differentiation for multilayer perceptrons**

Suppose each layer in our model now has optional parameters: $\theta_1,...\theta_4$
- $\mathcal{L}: R^n \to R$ so a scalar output
- Consider an $\mathcal{l_2}$ consider MLP with one hidden layer

$$
\mathcal{L}((\boldsymbol{x}, \boldsymbol{y}), \boldsymbol{\theta})=\frac{1}{2}\left\|\boldsymbol{y}-\mathbf{W}_2 \varphi\left(\mathbf{W}_1 \boldsymbol{x}\right)\right\|_2^2
$$
we can represent this as the following feedforward model:
- $\mathcal{L}=f_4 \circ f_3 \circ f_2 \circ f_1$
- $\boldsymbol{x}_2=\boldsymbol{f}_1\left(\boldsymbol{x}, \boldsymbol{\theta}_1\right)=\mathbf{W}_1 \boldsymbol{x}$
	- $W_1$: Weight Matrix
	- Weighted combinations of inputs
- $\boldsymbol{x}_3=\boldsymbol{f}_2\left(\boldsymbol{x}_2, \emptyset\right)=\varphi\left(\boldsymbol{x}_2\right)$
	- $\varphi$ is the activation function: ReLU
	- This is the activation Layer
	- Introduces Non-Linearity
- $\boldsymbol{x}_4=\boldsymbol{f}_3\left(\boldsymbol{x}_3, \boldsymbol{\theta}_3\right)=\mathbf{W}_2 \boldsymbol{x}_3$
	- Second Linear Layer
- $\mathcal{L}=\boldsymbol{f}_4\left(\boldsymbol{x}_4, \boldsymbol{y}\right)=\frac{1}{2}\left\|\boldsymbol{x}_4-\boldsymbol{y}\right\|^2$
	- Loss Function
	- Compares Prediction Against Target
It is Efficient to use Reverse mode differentiation to compute the gradients here. 

Computation of the Gratients
- $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_3}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_4} \frac{\partial \boldsymbol{x}_4}{\partial \boldsymbol{\theta}_3}$
- $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_2}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_4} \frac{\partial \boldsymbol{x}_4}{\partial \boldsymbol{x}_3} \frac{\partial \boldsymbol{x}_3}{\partial \boldsymbol{\theta}_2}$
- $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_1}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_4} \frac{\partial \boldsymbol{x}_4}{\partial \boldsymbol{x}_3} \frac{\partial \boldsymbol{x}_3}{\partial \boldsymbol{x}_2} \frac{\partial \boldsymbol{x}_2}{\partial \boldsymbol{\theta}_1}$
	- $d_k$ dimensional gradient row vector where $d_k$ is the number of parameters in layer $k$
	- Recursively computed by multipling gradient row vector at layer $k$ by the Jacobian $\frac{\partial \boldsymbol{x}_k}{\partial \boldsymbol{x}}$ which is an $n_k \times n_{k-1}$ matrix, where $n_k$ is the number of hidden units at layer $k$


**Cross Entropy Layer**

Setting: A cross Entropy Loss with Logit $x$ and targets labels $y$ as input returning a scalar:

$z=f(\boldsymbol{x})=$ CrossEntropyWithLogits $(\boldsymbol{y}, \boldsymbol{x})=-\sum_c y_c \log \left(\operatorname{softmax}(\boldsymbol{x})_c\right)=-\sum_c y_c \log p_c$
- $\boldsymbol{p}=\operatorname{softmax}(\boldsymbol{x})=\frac{e^{x_c}}{\sum^C}, e^{x^{\prime} c^{\prime}}$ are the predicted class probabilites,
- $y$ is true distribution over labels often one hot vecto
The Jacobian wrt to input would be:
- $\mathbf{J}=\frac{\partial z}{\partial \boldsymbol{x}}=(\boldsymbol{p}-\boldsymbol{y})^{\top} \in \mathbb{R}^{1 \times C}$
	- This is simply the graident with respect to the logits
		- Logits are not probabilites
		- Softmax turns them into probabilties
	- $z=f(\boldsymbol{x})=-\log \left(p_c\right)=-\log \left(\frac{e^{x_c}}{\sum_j e^{x_j}}\right)=\log \left(\sum_j e^{x_j}\right)-x_c$
		- Because $y$ is one-hot only one term survives
	- $\frac{\partial z}{\partial x_i}=\frac{\partial}{\partial x_i} \log \sum_j e^{x_j}-\frac{\partial}{\partial x_i} x_c=\frac{e^{x_i}}{\sum_j e^{x_j}}-\frac{\partial}{\partial x_i} x_c=p_i-\mathbb{I}(i=c)$
		- $log(a/b) = log(a) - log(b)$
		- Differeitatic $x_c$
			- Either -1 if $i=c$ or 0
			- Therefore can be written as indicator
	- Let $y = I(i=c)$

*Gradient for Softmax and Cross Entropy is just Predicted Probabilities -  True Probabilties*

For RELU
- The Subderivative is simply the Heaviside Step Function

**Computation Graphs**

MLPS are a kind of DNN in which each layer feeds directly into the next, forming a chain structure.
- By combining differentiable elements we are getting a Directed Acylic Graph where each node is a differentiable function of all its inputs

Example Function: $f\left(x_1, x_2\right)=x_2 e^{x_1} \sqrt{x_1+x_2 e^{x_1}}$
- Break this down into smaller parts
	- $\begin{aligned} & x_3=f_3\left(x_1\right)=e^{x_1} \\ & x_4=f_4\left(x_2, x_3\right)=x_2 x_3 \\ & x_5=f_5\left(x_1, x_4\right)=x_1+x_4 \\ & x_6=f_6\left(x_5\right)=\sqrt{x_5} \\ & x_7=f_7\left(x_4, x_6\right)=x_4 x_6\end{aligned}$
- For gradients we may need to sum gradients along multiple paths:
	- $\frac{\partial \boldsymbol{o}}{\partial \boldsymbol{x}_4}=\frac{\partial \boldsymbol{o}}{\partial \boldsymbol{x}_5} \frac{\partial \boldsymbol{x}_5}{\partial \boldsymbol{x}_4}+\frac{\partial \boldsymbol{o}}{\partial \boldsymbol{x}_7} \frac{\partial \boldsymbol{x}_7}{\partial \boldsymbol{x}_4}$
- Then we can avoid repeated computations by working in reverse topological order as follows:
	- $\frac{\partial \boldsymbol{o}}{\partial \boldsymbol{x}_j}=\sum_{k \in \mathrm{Ch}(j)} \frac{\partial \boldsymbol{o}}{\partial \boldsymbol{x}_k} \frac{\partial \boldsymbol{x}_k}{\partial \boldsymbol{x}_j}$
		- The sum over all children $k$ over node $j$
		- The gradient vector has already been computed for each child $k$ called adjoint
		- This gets multiplied by Jacobian of each child
A computation graph corresponding to an MLP with one hidden layer with weight decay. More precisely, the model computes the linear pre-activations , the hidden activations , the linear outputs , the loss , the regularizer $s=\frac{\lambda}{2}\left(\left\|\mathbf{W}^{(1)}\right\|_F^2+\left\|\mathbf{W}^{(2)}\right\|_F^2\right)$, and the total loss $J=L+s$.

### Training Neural Networks

Standard Approach is to use maximum likelihood estimation, by minimizing the NLL to train a NN:
$$
\mathcal{L}(\boldsymbol{\theta})=-\log p(\mathcal{D} \mid \boldsymbol{\theta})=-\sum_{n=1}^N \log p\left(\boldsymbol{y}_n \mid \boldsymbol{x}_n ; \boldsymbol{\theta}\right)
$$
- Just use the Backprop Algorithm to compute gradient of the loss and pass to ADAM optimizer
- DNN Loss is not a convex objective so in gneral we will not be able to find the global optimum
- SGD can find good solutions

**Tuning the Learning Rate**

Very Deep Models have an issue where the gradient tends to become either very small
- Vanishing Gradient Problem
or very big
- Exploding Gradient Problem
because errors gets amplified or diminished as it is passed through layers

Examples: Cosider gradient wrt a node at layer $l$
- $\frac{\partial \mathcal{L}}{\partial \boldsymbol{z}_l}=\frac{\partial \mathcal{L}}{\partial \boldsymbol{z}_{l+1}} \frac{\partial \boldsymbol{z}_{l+1}}{\partial \boldsymbol{z}_l}=\mathbf{J}_l \boldsymbol{g}_{l+1}$
	- if $J_l$ is constant across all layers, then the contribution of the gradient from the final layer $g_L$ to layer $L$ will be $J^{L-l} g_L$
		- Behavior depends on eigenvectors of $J$
			- $Jv = lambda v$
			- $J^2v = J(\lambda v) = \lambda^2 v$
				- Therefore repeated multiplicatons scales vectors by $\lambda^k$
			- Largest eigenvalue is the called the spectral radius $\rho(J)$
			- 
		- Multiplying by .5 for each layers can make the ending gradient very small for example, this is the vanishing gradient problem
		- Multplying by 2 repeatedely results in the gradient exploding
		- 
	- $J$ can have complex values 
		- Let $\lambda$ be the spectral radius of $J$ which is the maximum of the absolute values of the eigenvalues
		- If this is  $\lambda_{max}$ >1
			- gradients can explode
			- $\rho(J)^k$
			- 
		- Less than $\lambda_{max}$ <1
			- Gradient can vanish
		- Goal is to keep Jacobians near 1 for stability

Exploding Gradient Problem can be solved by gradient Clipping by using $$
\boldsymbol{g}^{\prime}=\min \left(1, \frac{c}{\|\boldsymbol{g}\|}\right) \boldsymbol{g}
$$
- Cap the magnitude of the gradient if it becomes too large
- $c$ is a threshold 
	- only the maginitude is changing

Various solutions to Solve the Vanishing Gradient Problem
- Modify activation functions at each layer to prevent the gradient from becoming too larger or too small
- Modify architecture so that the updates are additive rather than multiplicative
- Modify architecture to standardize the activations at each layer, so distributions of activations over the dataset remain constant during training
- Carefully choose intial parameters values

**Activation Functions that don't Saturate**

Problem with sigmoid
- The gradient signal is 0 when we saturate at 0 for large negative inputs and 1 for large positive inputs
- Backprop does not work then!!
- Gradient of sigmoid:
	- $\varphi^{\prime}(a)=\sigma(a)(1-\sigma(a))$
	- $\frac{\partial \boldsymbol{z}}{\partial \mathbf{W}}=\boldsymbol{z}(1-\boldsymbol{z}) \boldsymbol{x}^{\top}$
		- if z is near 0 or 1 then gradient wrt to parameters will be near 0

Common Non-Saturating Function

![[Common Activation Functions.png]]

1) $\operatorname{ReLU}(a)=\max (a, 0)=a \mathbb{I}(a>0)$
	1) Terms
		- Gradient: $\operatorname{ReLU}^{\prime}(a)=\mathbb{I}(a>0)$
		- Jacobian wrt inputs: $\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{x}}=\mathbf{W}^{\top} \mathbb{I}(\boldsymbol{z}>\mathbf{0})$
		- Jacobin wrt params: $\frac{\partial \boldsymbol{z}}{\partial \mathbf{W}}=\mathbb{I}(\boldsymbol{z}>\mathbf{0}) \boldsymbol{x}^{\top}$
	- Gradient Will not Vanish as long as $z$ is posiitve
	- Issues
		- If weights are too large and negative then $a = Wx$ takes on large negative values so $z=0$
			- Then Gradient will go to 0
			- So hidden units will stay off permanently
			- Called "dead ReLU" problem

2) Non Saturating ReLU: 
	1) Leaky Relu: $\operatorname{LReLU}(a ; \alpha)=\max (\alpha a, a)$
		1) Slop is $1$ for positive inputs
		2) Allow $\alpha$ to be learned then we get parametric ReLU
	2) ELU:$\operatorname{ELU}(a ; \alpha)= \begin{cases}\alpha\left(e^a-1\right) & \text { if } a \leq 0 \\ a & \text { if } a>0\end{cases}$
		1) Advantage over leaky: smooth function
	3) SELU:$\operatorname{SELU}(a ; \alpha, \lambda)=\lambda \operatorname{ELU}(a ; \alpha)$
		1) By choosing $\alpha$ and $\lambda$ to be carefully chosen values then the activation is guaranteed to ensure that output of each layer is standardized
3) Other Choices
	1) Swish: Image Classification
	2) SiLU
	3) GELU
	4) SmelU

**Residual Connections**

Solving Vanishing Gradient Problems:
- Use Residual Network
- Each layer has the form of a residual block defined by: $\mathcal{F}_l^{\prime}(\boldsymbol{x})=\mathcal{F}_l(\boldsymbol{x})+\boldsymbol{x}$
	- $F_l$ is a standard shallow nonlinear mapping
	- Inner function computed residual that needs to be added to compute desired output
- Gradient at layer $l$ depends directly on the gradient at layer L in a way that is indepdent of the depth of the network

**Initialization Schemes**

Each layer we have a weights matrix: $W =\in R^{n_{out} \times n_{in}}$
- $W_{i j} \sim \mathcal{N}\left(0, \sigma^2\right)$
- We are setting the scale of the randomness
We are choosing starting energy level of signals in the network

$z = Wx$
- if $x_{ij}$ has variance 1 then wegihts are random with variance $\sigma^2$
- Then $\operatorname{Var}\left(z_i\right) \approx n_{\mathrm{in}} \cdot \sigma^2 \cdot \operatorname{Var}(x)$
- We want the Var(z) = Var(x)

```
W = np.random.randn(n_out, n_in) * sqrt(2/n_in)
or 
torch.nn.init.xavier_normal_(W)
torch.nn.init.kaiming_normal_(W)
```

Do not
- Sample from standard normal with fixed variance: results in exploding gradients
Use
- Xavier, Glorot: $\sigma^2=\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}$
	- Sigmoid, tanh, that balances forward and back variance
- He: $\sigma^2=2 / n_{\mathrm{in}}$
	- ReLU, Leaky Relu
- LeCun: $\sigma^2=1 / n_{\text {in }} ;$
	- SeLU

We wan't to ensure and satisfy that : $\frac{1}{2}\left(n_{\mathrm{in}}+n_{\mathrm{out}}\right) \sigma^2=1$,
- $n_{in}$ number of incoming connections
	- affects forward signal size
- $n_{out}$ number of outgoing connections
- Scale the weights so that total variance stays stable across forward and backward passes

Alternative Methods:
- layer-sequential unit-variance
	- This scheme can be viewed as an orthonormal initialization combined with batch normalization performed only on the first mini-batch.


### Regularization

How to avoid overitting? 
- Early Stopping: stopping the training perocedure when the error on training starts to increase
- Weight Decay: Impose a prior on parameters, then use MAP Estimation
	- W: $\mathcal{N}\left(\boldsymbol{w} \mid \mathbf{0}, \alpha^2 \mathbf{I}\right)$
	- Bias: $\mathcal{N}\left(\boldsymbol{b} \mid \mathbf{0}, \beta^2 \mathbf{I}\right)$
	- Equivalent to L2 regularization of the objective encouragin small weights
- L1 Regularization or ARD
	- Sparse matrixes are hardley helpful for current GPUs
- Dropout
	- Randomly turn off outgoing connections with probability $p$
	- Prevents complex co-adaptation of the hidden units
	- Each units must learn to perfrom well even if some of the other units are missing at random
	- Can be viewed as estimating a noisy version of the wegiths, where we have a Bernouli noise term
	- If we do this during Testing then we have an ensemble of networks , each with a slightly different sparse graph structurec called Monte Carlo Dropout
		- $p(\boldsymbol{y} \mid \boldsymbol{x}, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S p\left(\boldsymbol{y} \mid \boldsymbol{x}, \hat{\mathbf{W}} \epsilon^s+\hat{\boldsymbol{b}}\right)$
			- $S$ is the number of samples,
			- $\hat{\mathbf{W}} \epsilon^s$ to indicate that we are multiplying all the estimated weight matrices by a sampled noise vector
- Bayesian Neural Networks
	- When we have more parameters then data points, multiply models can fit data very well
	- Capture the induced uncertaintity in the poster by margenlizing out parameters by computing $p(\boldsymbol{y} \mid \boldsymbol{x}, \mathcal{D})=\int p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta}$
	- Infinite ensemble of differently weighted neural networks
- Regularization Effects on SGD
	- Sharp Minima: Deep holes in the loss
		- Solutions correspond to results that are overfitting the data
	- Flat Minima:
		- More robust and generalize well
		- regions in parameter space where there is a lot of posterior uncertainty, and hence samples from this region are less able to precisely memorize irrelevant details about the training set
		- nds such flat minima by virtue of the addition of noise, which prevents it from “entering” narrow regions of the loss landscape: Called Implicity Regularization
	- Flat Minima: This can be encouraged by using entropy SGD or sharpness aware minimalization, stochastic weight averaging
		- SGD has computational and statistical advantages
- Over Parametrized Models
	- Small model underfit, medium models are a good fit, and large models overfit
	- Moden Deep Neural Networks have $P >> N$
		- Classic: Overfit
		- Reality: Training error 0 and test error also becomes good
		- Model has many possible intropolating solutions
			- it can choose smooth ones
		- Optimizers like ADAM or SGD have a bias toward
			- Smooth
			- Low Complexity
			- Lowe narm
			- Called Implicit Regularization

### Other Kinds of Feedforward Networks

Consider a 1 layer neural net where the hidden layer is given by feature  vector: $$
\phi(\boldsymbol{x})=\left[\mathcal{K}\left(\boldsymbol{x}, \boldsymbol{\mu}_1\right), \ldots, \mathcal{K}\left(\boldsymbol{x}, \boldsymbol{\mu}_K\right)\right]
$$
- $\boldsymbol{\mu}_k \in \mathcal{X}$ are a set of $K$ centroids or exemplars
- $\mathcal{K}(\boldsymbol{x}, \boldsymbol{\mu}) \geq 0$ is a kernel function
- Gaussian Kernel: $\mathcal{K}_{\text {gauss }}(\boldsymbol{x}, \boldsymbol{c}) \triangleq \exp \left(-\frac{1}{2 \sigma^2}\|\boldsymbol{c}-\boldsymbol{x}\|_2^2\right)$
	- $\sigma$ : bandwidth of the kernel
		- Invariant
		- only a function of the distaince: $r=\|\boldsymbol{x}-\boldsymbol{c}\|_2$
		- $\mathcal{K}_{\text {gauss }}(r) \triangleq \exp \left(-\frac{1}{2 \sigma^2} r^2\right)$
- Called a Radial Basis Function Kernel or RBF Kernel
- 1 Layer Neural Netwrok as the hidden layer with RBF Kernels is called an RBF Network: $p(y \mid \boldsymbol{x} ; \boldsymbol{\theta})=p\left(y \mid \boldsymbol{w}^{\top} \boldsymbol{\phi}(\boldsymbol{x})\right)$
	- where $\boldsymbol{\theta}=(\boldsymbol{\mu}, \boldsymbol{w})$
	- Centroids: $\mu$ are fixed then solve for optimal weights $w$
	
- RBF Network for regression
	- $p(y \mid \boldsymbol{x}, \boldsymbol{\theta})=\mathcal{N}\left(\boldsymbol{w}^T \boldsymbol{\phi}(\boldsymbol{x}), \sigma^2\right)$
- RBF Network for classification
	- $p(y \mid \boldsymbol{x}, \boldsymbol{\theta})=\operatorname{Ber}\left(\sigma\left(\boldsymbol{w}^T \boldsymbol{\phi}(\boldsymbol{x})\right)\right)$
- Mixtures of experts
	- Common to assume a unimodal output distribution such as Gaussian or Student Distribution, where the mean and variance is some function of the input: $p(\boldsymbol{y} \mid \boldsymbol{x})=\mathcal{N}\left(\boldsymbol{y} \mid f_\mu(\boldsymbol{x}), \operatorname{diag}\left(\sigma_{+}\left(f_\sigma(\boldsymbol{x})\right)\right)\right)$ where $f$ functions may be MLPs
		- This one-to-many functions
		- 