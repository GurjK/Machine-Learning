# Contents
1. Standardizing and whitening data
2. Transformations
	1. Box Cox Tranformation
	2. Logit
	3. Scaling
	4. Data Smoothing
3. Method for 1 to many transformations
	1. Basis Function 
	2. Splines
	3. MARS (Multivariate Adaptive Regression Splines)
	4. Binning
4. Methods for many to many transformations
5. Autoencoders
6. Spatial Sign
7. Distance and Depth Features
8. Profile Data


## Notes
- Trees are immune to predictor distributions that are skewed or to individual samples that have unusual value
- K Nearest Neighbors and SVM are more sensitive to predictors with skewed distributions or outliers
- Partial least squares, for instance, is specifically built to directly handle highly correlated predictors. But models like multiple linear regression or neural networks are adversely affected in this situation.
- 

### 1) Standardizing and whitening data

For a dataset, $X \in R^{ N \times D}$ it is common to preprocess the data so that each column has zero mean and unit variance.
- Called **Standardizing Data**
	- K nearest Neighbors, SVM, Lasso, Ridge require standardization

But this does not remove the correlation between the columns, to do this we must **whiten** the data.

Let us define the empirical covariance matrix as follows: be $\boldsymbol{\Sigma}=\frac{1}{N} \mathbf{X}^{\top} \mathbf{X}$, and let $\boldsymbol{\Sigma}=\mathbf{E D E}^{\top}$ be its diagonalization. Equivalently, let $[\mathbf{U}, \mathbf{S}, \mathbf{V}]$ be the SVD of $\frac{1}{\sqrt{N}} \mathbf{X}$ (so $\mathbf{E}=\mathbf{V}$ and $\mathbf{D}=\mathbf{S}^2$, as we discuss in Section 20.1.3.3.) Now define

$$
\begin{equation*}
\mathbf{W}_{p c a}=\mathbf{D}^{-\frac{1}{2}} \mathbf{E}^{\top} 
\end{equation*}
$$
- This is the **PCA Whitening Matrix**
	- Not unique: Any rotation of it will maintain the whitening property
	- Let $y = W_{pca}x$ then $\operatorname{Cov}[\boldsymbol{y}]=\mathbf{W} \mathbb{E}\left[\boldsymbol{x} \boldsymbol{x}^{\top}\right] \mathbf{W}^{\top}=\mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{\top}=\left(\mathbf{D}^{-\frac{1}{2}} \mathbf{E}^{\top}\right)\left(\mathbf{E D E}^{\top}\right)\left(\mathbf{E D}^{-\frac{1}{2}}\right)=\mathbf{I}$

Let $R =E$: then we get $$\mathbf{W}_{z c a}=\mathbf{E D}^{-\frac{1}{2}} \mathbf{E}^{\top}=\mathbf{\Sigma}^{-\frac{1}{2}}=\mathbf{V S}^{-1} \mathbf{V}^{\top}$$
- This is called the **Mahalanobis whitening or ZCA**
	- Advantage over PCA whitening is that result transformed data is as close as possible to original data in least square sense!
	- ![[ZCA.png]]

## 2) Transformations 

**Box Cox Transformation**
- Uses MLE to estimation transfomation parameter $\lambda$ in equation: $x^*= \begin{cases}\frac{x^\lambda-1}{\lambda \tilde{x}^{\lambda-1}}, & \lambda \neq 0 \\ \tilde{x} \log x, & \lambda=0\end{cases}$
	- $\tilde{x}$ is the geometric mean of the predictor data.
	- This is called a power transfomation
		- $\lambda=1$ : No Transformation
		- $\lambda=0.5$ : Square Root
		- $\lambda=0$ : Log
		- $\lambda=-1$ : Inverse
	- Can only be applied to data that is strictly positive
		- **Yeo and Johnson** for negative data too!
	- Unsupervised
		- might improve predictor distribution but no guarantee it improves model
		- but in linear, neural networks, and svm skewness can dominate calculations so it helps

**Logit**
- Converts data on a scale from 0 to 1 to negative and positive infinity
	- When model outcomes are not in same boundary
	- inverse logit can be use to place back on original scale
- $\operatorname{logit}(\pi)=\log \left(\frac{\pi}{1-\pi}\right)$
	- arcsin is alternative: $y^*=\operatorname{arcsine}(\sqrt{\pi})$

**Scaling**
- Process of dividing a variable by the corresponding training set's standard deviation
- Ensures that variables have a standard deviation of 1
	- Must be estimated from training and applied to both training and test set!
- Needed when distance or dot products are used or when variables are required to be on a comoon scale to apply a penalty correctly
	- K nearest Neighbors, SVM, Lasso, Ridge

**Smoothing**
- For time series data we can smooth the data using the following to eliminate noise
	- Average of mean or median (Moving average or Moving median with short term range)
	- Smoothing splines


## 3) Method for 1 to many transformation

Purpose: Create multiple indicator columns from a single qualitative predictor. Similarly, transformations can be made on a single numeric predictor to expand it to many predictors.

**Nonlinear Features via Basis Expansions and Splines**
- A basis expansion of a predictor $x$ can be achieved by deriving a set of function $f_i(x)$ that can be combined using a linear combination. 
- Cubic Basis Function: $f(x)=\sum_{i=1}^3 \beta_i f_i(x)=\beta_1 x+\beta_2 x^2+\beta_3 x^3$
	- The original columns $x$ is turned into two new features $x^2$ and $x^3$
	- $\beta$ is calculated by linear regression
		- Could be estimated in presdence of other regressors
		- If truly linear $\beta_2$ and $\beta_3$ would be 0
- Polynomial Spline
	- Basis expansion created different regions of the predictor space where boundaries are called knots
	- Uses polynomial functions within regions to represent data
	- number of knots controls number of regions and complexity
		- low: basis function can rpresent simple trends
	- natural cubic spline is most common
		- First knot is usally taken at the intercept
	- How many knots?
		- Tuning parameter determined using grid search
		- generalized cross validation
	- Created at priori to being exposed to outcome data but does not need to be the case
		- Asssing every training set point as a potential knot and uses regularized regression models to determine which instances should be knots
		- Smoothing spline Mehtodology
		- GAMS
			- adapteively model separate basis functions for different variables and estimate the complexity for each
			- different predictors can be modelled with different levels of complexity
		- Example:
			- x=7.3 would become 0,.12,.73,.15,0
- MARS (Multivariate Adaptive Regression Splines)
	- Single fixed, knot spline
	- Hinge function transformed by: $h(x)=x I(x>0)$
		- $h(x-3.75)$: gets a new feature that is zero when log lot area is less than 375 and $x-3.75$ otherwise
		- Create a 
		- Alos known as $ReLu$
		- While basis functions can be effective at helping build better models by representing nonlinear patterns for a feature, they can be very effective for exploratory data analysis.
- Binning (categorization or discretization)
	- translating a quantitative variable into a set of two or more qualitative buckets
		- Ex: use 4 quantiles to split up data into 4 distinct values
	- Reasons to use this?
		- Variable like age lets you say that there is a 25% increase in probability of the event between young and old people
		- Avoids speciying relationship between predictor and outcome
		- Gives perception of reducing variation in the data
	- Issues
		- Unlikely that underlying trend is consistent with data
		- Makes it harder for model to do an effective job since all the nuance in data has been removed
		- no rationale for cutoff point
		- Substantial increase in  probability that erroneous trend will be discovered if there was no trend intiially
	- Overall this is a last resort and determination of the bins must be included in the resampling. 

## 4) Methods for many to many transformations

**Linear Projection Methods**
- Methods that take a matrix of numeric predictor values $X$ and create new components that are linear combinations of the original data called *scores*
- $n$ tranining set points and $p$ predictors, the scored could be denoted as $n \times p$ , $X^* = XA$ 
	- $A$  is a projection matrix that is $p \times p$
	- $x_{i 1}^*=a_{11} x_{i 1}+a_{21} x_{i 2}+\ldots+a_{p 1} x_{i p}$
- Idea: use a protion of $A$ such that $k<p$ such to reduce the dimensions of the problem
	- This is not feature selection
	- This is a function of the original predictors
	- Predictor redundancy can be reduced
		- But since it is unsupervised it does not guarantee an improvement in prediction
	- Supervised methods improve on this by using a subspace that uses outcome but need more data to determine this

1. PCA (Principal Component Analysis)
	1. Unsupervised
	2. Objective: Find a linear combination of the original predictors such that the combinations summarize the maximal amount of variation (information) in the original predictor subspace so we are finding subspace that contains the information relevant to predictors
		1. PCA scores are required to be orthogonal (uncorrelated)
			1. Enables predictors space variability to be neately partitioned in a way that does not overlap
			2. PCA scores are uncorrelated
				1. Good for multiple linear regression, neural networks, support vector machines and others that need predictors to be uncorrelated
		2. In practice , enough new score variables are computed to account for say 95% of the original variability
	3. Math
		1. Variance summarization can be solved by finding the eigenvalues and eigenvectrs of the covariance matrix of the predictors. 
	4. When is it useful?
		1. Data composed of one or more clusters of predictors that contain redundant information (highly correlated)
			1. They can exist in a lower dimensional space then original data
			2. Ex: draw a line through two predictors plotted, the points on the line are orthogonal projections of the original data onto first principal component score , then residual would be the second score
		2. When predictors are _linearly correlated_ and when the resulting scores are associated with the response.
	5. Useful techniques
		1. Plot the first few scores aginst one another to look for separation between classes, outliers, or other patterns.
		2. Biplot, overlays the scores along with the coefficients of $A$
			1. Common for the scale of the scores to become smaller and smaller, so plot the compoent of the scores on a common axis scale  to account for all of the data in the scores
2. Kernel PCA
	1. Unsupervised
	2. Objective: In PCA, orthogonal projecting may not provide a good relationship with the response, especially if a non-linear relationship. The kernel PCA approach combines a specific mathematical view of PCA with _kernel functions_ and the kernel ‘trick’ to enable PCA to expand the dimension of the predictor space in which dimension reduction is performed (not unlike basis expansions).
		1. Ex: Say model is $y=x_1+x_1^2+x_2+x_2^2+\epsilon$ . PCA would capture $x_1$ and $x_2$ and ignore quadratic part. Could add this to predictor after but could use a kernel basis to do this instead
		2. Instead of performing a non-linear transformation, we use the inner products only to estimate the kernel
			1. This is a similarity measure in transformed space
		3. Steps:
			1. Comute kernel matrix: which gives a similarity matrix
			2. Center the matrix
			3. Perform eigen decomposition on kernel matrix
			4. Coordinates along principal directions become new engineered features
	3. Math
		1. The eigen Decomposition of the Covariance Matrix can be equivalently written in terms of the inner products of the samples. 
			1. Known as Dual representation of the problem.
			2. Inner product is know as linear kernel
			3. Linear kernel of $x_1$ and $x_2$ is $k\left(\mathbf{x}_1, \mathbf{x}_2\right)=\left\langle\mathbf{x}_1, \mathbf{x}_2\right\rangle=\mathbf{x}_1^{\prime} \mathbf{x}_2$
				1. For a set of $i=1 \ldots n$ samples and two predictors, this corresponds to $k\left(\mathbf{x}_1, \mathbf{x}_2\right)=x_{11} x_{12}+\ldots+x_{n 1} x_{n 2}$
		2. Kernel PCA uses the kernel substituion trick to exchange linear kernel with any other valid kernel. Some of the kernels are
			1. Basic Polynomial: $k\left(\mathbf{x}_1, \mathbf{x}_2\right)=\left\langle\mathbf{x}_1, \mathbf{x}_2\right\rangle^d$
				1. Let $d=2$ and 2 features then: $k\left(\mathbf{x}_1, \mathbf{x}_2\right)=\left(x_{11} x_{12}+\ldots+x_{n 1} x_{n 2}\right)^2$
			2. Radial Basis Function (Gaussian): $k\left(\mathbf{x}_1, \mathbf{x}_2\right)=\exp \left(-\frac{\left\|\mathbf{x}_1-\mathbf{x}_2\right\|^2}{2 \sigma^2}\right)$
	4. When is it useful?
		1. When there is not a linear relationship between predictors and the data. 
			1. Ex: Data is a moon, circle, spiral, curved surface cause regular PCA would not capture the structure well
	5. Useful Plots?
3. ICA (Independent Component Analysis)
	1. Unsupervised
	2. Objective: PCA means that the directions will not be correlated (linearly related) with each other but they may be  not be independent (covariance =0). ICA creates a linear relationship combination but it ensures that they are statistically indepdent from one another as possible. Allows it to model a broader set of trends
		1. Ways to meet this: 
			1. Often maximize non-gaussianity of resulting components
			2. fastICA: From information Theory measured by negentropy
		2. Realistically should capture methods that are dissimilar from PCA
			1. Won't happen if significant multivariate normality or strictly linear trends
		3. Common to perform the following
			1. Normalize and Whiten the data
				1. Whitening means to convert the original values to the full set of PCA components
				2. ICA is s
	3. Math
	4. When is it Useful
		1. Note: ICA is sensitive to initial random values it is set to! Make sure to set these manually to reproduce results
		2. 
4. NNMF (Non negative matrix factorization)
	1. Unsupervised
	2. Objective: Specific to features that are greater than or equal to 0. Find coefficients of $A$ so that values are also nonnegative. Find scores as close as possible to original data with the constaint of non-negativity.
		1. Closeness can be measured by
			1. KL divergence
			2. Mean Squared Error
	3. Math:
	4. When is it useful:
		1. This approach is popular for text data where predictors are word counts, imaging, and biological measures (e.g., the amount of RNA expressed for a specific gene).
		2. Notes: Like ICA, the numerical solver is initialized using random numbers and the final solution can be sensitive to these values and the order of the components is arbitrary.
5. PLS (Partial Least Squares) 
	1. Supervised
	2. Objective: Reduce dimension using PCA such that it is optimally related to the response. Find linear functions (latent variables ) of the predictors that have optimal covariance with the response
		1. Ensure that scores has highest possible covariance with the response
	3. Math
	4. When is it Useful
		1. Notes: Requires that data in each of new dimensions is uncorrelated
		2. PLS can more effectively reduce dimensions


### 5) Autoencoders
- Complex Multivariate methods for finding representations of predictor data and commonly used in deep learning
	- Create nonlinear mapp between original and set of ariticial features 
	- Used to model predictions
- All of predictor data can be used to create an equation that estimates the relationships between the predictors
	- Based on patterns, a smaller data set of predictors can be more effectively represented by using their predicted feature values
- Can be created by a neural network
	- Predictor -> Hidden Layer 1 -> Hidden layer H -> Feature
	- Between these steps are nonlinear functions (ReLu)
	- Connvection to output is linear
	- Using a small number of nodes forces that autoencoder to learn patterns in the data
	- Beneficial when you have a large amount of data, overfitting controlled by regularization and dropout
- It is common to treat the relationships estimated by the autoencoder model as “known” values and not re-estimate them during resampling.
	- 

- When to use?
	- abundance of unlabeled data(outcome is not known)
	- In OLS, Estimating $(X^TX)^{-1}$


## 6) Spatial Sign
- Takes a set of predictor values and transforms them in a way that the new values have the same distiance to the center of the distribution: $x_{i j}^*=\frac{x_{i j}}{\sum_{j=1}^p x_{i j}^2}$
	- Global constant normalization
	- Predictors need to be centered and scaled prior to the spatial sign since computing norm!
	- If variables are highly skewed, then possibly a transformation to induce symmetry may help like the Yeo-Johnson technique
	- mitigates damage of extreme outliers

## 7) Distance and Depth Features
- Classification: beneficial to make semi-supervised features from the data
	- Outcome classes are used in creation of new predictors but no optimized for predictive accuracy
	- Ex: compute class centroids
		- New Data uses distance to class centroids and use this as new features
		- quasi nearest neighbors technmique
- Data Sepht: measures how close a data point is to the center of its distribution
	- Inverse relationship between class distances and data septh
	- Large depth values indicate close to class center

## 8) Profile Data

Profile Data: This type of data can occur if a sample is measured repeatedly over time, if a sample has many highly related/correlated predictors, or if sample measurements occur through a hierarchical structure.

Say we have hourly data that needs to be converted to Daily to match the scale of the outcome variable
- mean/median/rate of change

To remove background drift from data commonly used in signal processing and spectroscopy. You measured data that contains both the real signal you care about and an unwanted background trend(baseline). Measured Data = True signal + Baseline+Noise
- Points above fit are signal peaks and below are baseline points
	- Algo fits a curve, removes upper points, and refits lower points
- Select the polynomial degree, , and maximum number of iterations, ;

	2:while external resample do

	3:Fit a polynomial of degree  to the data

	4:Calculate the residuals (observed minus fitted)

	5:Subset the data to those points with negative residuals

	6:end while

	7:Subtract the estimated baseline from the original profile

To remove Extrenous Noise. Raw Spectra are messy, inconsistent, noisy, and difficult to compare directly.
- Need to clean data to get true chemical information 
- irrelevant variation is removed

Use standard Normal Variate to center and scale this correctly
- Issue: outliers can distort the mean and standard deviation
- Ignore extreme values when computing mean/std
	- remove top and bottom 5%
To fix jagged noise (random measurement noise)
- replace each point with average of nearby points

To reduce correlation
- PCA
- first order differentaition, the response at the $(p-1)^st$ value in the profile is subtracted from the reponse at the $pth$
	- difference represents the rate of change of the response btween consecutive measurements in the profile!
	- Autocorrelation is reduced!

![[Profile Data transformations.png]]