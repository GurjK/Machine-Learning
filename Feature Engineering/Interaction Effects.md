Additional variation in the response can be explain by the effect of two or more predictors working in conjunction with each other.
- Ex: Crop yield needs both water and fertilizer to grow

More formally, two or more predictors are said to interact if their combined effect is different (less or greater) than what we would expect if we were to add the impact of each of their effects when considered alone. Note that interactions, by definition, are always in the context of _how predictors relate to the outcome_.
- Correlations between predictors, for example, are not directly related to whether there is an interaction effect or not. Also, from a notational standpoint, the individual variables (e.g., fertilizer and water) are referred to as the _main effect_ terms when outside of an interaction.

Near Parallel lines when plotted display no interaction effect but cross line do!


$$
y=\beta_0+\beta_1 x_1+\beta_2 x_2+\beta_3 x_1 x_2+\text { error }
$$
- In this equation, $\beta_0$ represents the overall average response
- $\beta_1$ and $\beta_2$ represent the average rate of change due to $x_1$ and $x_2$
- $\beta_3$ represents the incremental rate of change due to the combined effect of $x_1$ and $x_2$ that goes beyond what $x_1$ and $x_2$ can explain alone. 

4 Possible Case Scenarios Cane Arise from this model.

1. $\beta_3$ is not significant, so no interaction exists and $x_1$ and $x_2$ are additive
2. $\beta_3$ is negative, $x_1$ and $x_2$ alone also affect the reponse, then interaction is called antagonstic
3. $\beta_3$ is positive, , $x_1$ and $x_2$ alone also affect the reponse, then interaction is called synergystic
4. $\beta_3 \neq 0$ but one or both $x_1$ and $x_2$ do not affect the response. In this case, the average response of  across the values of  (or vice versa) has a rate of change that is essentially zero. However, the average value of  at each value of  (or conditioned on ) is different from zero. This situation occurs rarely in real data, and we need to understand the visual cues for identifying this particular case. This type of interaction will be called _atypical_.

Notes
- Tree based model inherently model interacctions between predictors through subsequent recursive splits in the tree
	- RF too!

**Guiding Principles**
- Expert knowledge of the system under study is critical
- Hierarchy principle: higher degree of the interaction, the less likely it would explain variation in reponse
	- Ex: first order: $x_1$, second order: $x_1x_2$, third order: $x_1x_2x_3$
	- three way and beyond are not likely to capture a significatn amount of variation and can't be interpretted
- Effect Sparsity: Only a fraction of possible effects truly explain a significant amount of variation reponse
- Heredity Principle: Interactions terms may only be considered if the ordered terms preciding the interaction are effective at explaining repsonse variation
	- Strong: interaction to be considered , all lower-level terms must explain a significant amount of response variation
	- Weak: Any possible interaction with the significant factor
- Order of preprocessing and interactions affects results
	- Interaction terms should probably be created prior to any preprocessing steps


### **Brut Force Approach to Identifying Predictive Interactions**
1. Simple Screening
2. Penalized Regression

**Simple Screening**

Use nested statistical models, so for linear regression first run the main effect model and then the main effects plus interactions: 
1. $y=\beta_0+\beta_1 x_1+\beta_2 x_2+$ error
2. $y=\beta_0+\beta_1 x_1+\beta_2 x_2+\beta_3 x_1 x_2+$ error

For linear regression
- Compare the residual error between the two mode
- Run hypothesis test evaluates where the improvement in error, is considered real

Alternate Approach
- Resampling is used to create a number of nested model pairs for different version of the training sets
	- Assesment set is sused to evaluate objective function
	- Any measure of performance can be used to compare the nested model

Both approaches calculate p values
- Do not overoptimize for these
- Alternative to p value is Bonferroi correction, which uses a very strict exponential penality to minimize false postive findings
- 3 p value approaches
	- Regular p value
	- FDR p value
	- Bonferroni adjustment

Notes
- Traditional approach repredicts on the training data so overfitting can happen!
- significant findings are result of overfitting the interactions to the data
- This is best when complete enumeration is possible


**Penalized Regression**

Issue with simple screening: However, evaluating interaction terms in a one-at-a-time fashion prevents the simplistic models from evaluating the importance of the interaction terms in the presence of the full model the individual predictors and pairwise interaction terms.

Another Approach: Create and add all interactions terms to the data. Issue is that we have more predictors than samples now. 

Penalized models are for when we have more predictors then samples. 
- Main issue from Linear Regression: Need to inverse the covariance matrix
- Soo we instead have Lasso and Ridge

Lasso: $S S E_{L_1}=\sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2+\lambda_{\ell} \sum_{j=1}^p\left|\beta_j\right|$
- Eliminating Predictors
Ridge: $S S E_{L_2}=\sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2+\lambda_r \sum_{j=1}^P \beta_j^2$
- Collinearity
GLM: $S S E=\sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2+(1-\alpha) \lambda_r \sum_{j=1}^P \beta_j^2+\alpha \lambda_{\ell} \sum_{j=1}^P\left|\beta_j\right|$
- $\alpha = 1$ means a 100% lasso model

### **Approaches when Complete Enumeration isn't possible**

1. Two Stage Modelling
2. Tree Based Methods
3. The Feasible Solution Algorithm

**Two Stage Modelling**

Not very feasible but can use glmnet mode maybe

**Tree Based Methods**

Recursive partitioning
- Find the optimal split point of a predictor that separates the samples into more homegeneous groups with respect to the response
	- Repeated until minimum number of samples in a node is met or process mets otpimization criterion
- In essence, the partitioning aspect of trees impedes their ability to represent _smooth, global interactions_. which is why tree based methods are so large to determine simple synergestic relationships
- Random Forest and variations of it can be used to find high order interaction effects!!
- Why are trees good for this?
	- They are aggregating many trees from slightly different versions of the original data
	- trees are good when there is a local space where an intraction works, trees are good here!
- Best Method: Feature Weighted Random Forests
	- randomly selects features based on weights determined by feature's importance
	- After the ensemble is created, a metric for each co-occurring set of features is calculated which can then be used to identify the top interacting features.
- Other Method: Partial Dependence
	- $H$ Statistic: In short, this approach compares the joint effect of two (or more) predictors with the individual effect of each predictor in a model. If the individual predictor does not interact with any of the other predictors, the difference between the joint effect and the individual effect will be close to zero. However, if there is an interaction between an individual predictor and at least one of the predictors, the difference will be greater than zero.
- Other Method: 
	- If if an interaction among predictors is important, then a tree-based method will use these predictors multiple times throughout multiple trees to uncover the relationship with the response. These predictors will then have non-zero, and perhaps high, importance values. Pairwise interactions among the most important predictors can then be created and evaluated for their relevance in predicting the response.

**If meaningful predictor relationships can be identified and placed in a more effective form, then a simpler and perhaps more appropriate model to estimate the relationship between the predictors and the response can be used.**


**Feasible Solution Algo**

Substitution Algorithm!
- Randomly select 3 out of 10 variables
	- Measure performance
- Now fix second 2 and measure other 7
	- If none of the other seven predictors give better performance then $x_9$ then this is kept!
	- If it is not optimal then it is exchanged for another
- Then fixes the first and third predictors

This is called FSA algo.  Lambert’s approach begins with a base model, which typically includes predictors thought to be important for predicting the response. The order of the desired interaction is chosen (i.e., two-factor, three-factor, etc.). The process of identifying potentially relevant interactions follows the same type of logic as the original FSA.

Overview: base model included three predictors ( $x_1, x_5$, and $x_9$ ) and the goal is to identify important pairwise interactions. Lambert's algorithm randomly selects two predictors, say $x_3$ and $x_9$, and computes the performance of the model that includes the base terms and this interaction term. The algorithm then fixes the first predictor and systematically replaces the second predictor with the remaining eight (or nine if we want to study quadratic effects) predictors and computes the corresponding model performance. If none of the other predictors give better predictive performance, then $x_9$ is kept. But if $x_9$ is not optimal, then it is exchanged with the best predictor. The process then fixes the second predictor and exchanges the first predictor with eight (or nine) of the other predictors. This process continues to iterate until convergence. Then a different random pair of predictors is selected and the process is applied. The converged interactions are tallied across the random starts to identify the potential feasible solutions. This algorithm can be easily generalized to any order of interaction.

Other Possible Interaction Models
- MARS
- Cubist


