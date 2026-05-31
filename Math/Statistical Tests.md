## Evaluation Metrics

$\mathrm{RSS}=e_1^2+e_2^2+\cdots+e_n^2$ 
- $\operatorname{RSS}=\left(y_1-\hat{\beta}_0-\hat{\beta}_1 x_1\right)^2+\left(y_2-\hat{\beta}_0-\hat{\beta}_1 x_2\right)^2+\cdots+\left(y_n-\hat{\beta}_0-\hat{\beta}_1 x_n\right)^2$
- RSS measures the amount of variability that is left unexplained after performing the regression.

$\operatorname{SE}\left(\hat{\beta}_0\right)^2=\sigma^2\left[\frac{1}{n}+\frac{\bar{x}^2}{\sum_{i=1}^n\left(x_i-\bar{x}\right)^2}\right], \quad \operatorname{SE}\left(\hat{\beta}_1\right)^2=\frac{\sigma^2}{\sum_{i=1}^n\left(x_i-\bar{x}\right)^2}$
- $\sigma = Var{(\epsilon)}$
- Used to compute Confidence Intervals

$\operatorname{Var}(\hat{\mu})=\operatorname{SE}(\hat{\mu})^2=\frac{\sigma^2}{n}$ 

- Assume that errors $\epsilon$ for each observation is $\sigma^2$ and they are uncorrelated

Residual Standard Error
- $\operatorname{RSE}=\sqrt{\operatorname{RSS} /(n-2)} =\sqrt{\frac{1}{n-2} \sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2} .$
- Estimate of the standard deivation of $\epsilon$
- Average amount that the response will deivate from teh true regression line
- Considered lack of fit of the model to the true data
- Goal: Should be small indicating that the data fits the model well!

R-Squared ($R^2$)
- $R^2=\frac{\mathrm{TSS}-\mathrm{RSS}}{\mathrm{TSS}}=1-\frac{\mathrm{RSS}}{\mathrm{TSS}}$
	- $\mathrm{TSS}=\sum\left(y_i-\bar{y}\right)^2 \text { is the total sum of squares }$
	- TSS measures total variance in response variable
	- RSS measures the amount of variability that is left unexplained after performing the regression.
	- so TSS - RSS = measures amount of variability that is explained by performed in the regression
- Proportion of variance explained 
- measures the proportion of variability in Y that can be explained using X.
- In univariate Setting: $R^2 = Correlation^2$
- Goal: Should be close to 1

Confidence Intervals
- Range of vlaues that with 95% probability the range will contain the true unknown value of the parameter
- if we take repeated samples and construct the confidence interval for each sample, 95% of the intervals will contain the true unknown value of the parameter.

$CI(\beta_1)$ = $\hat{\beta}_1 \pm 1.96 \cdot \mathrm{SE}\left(\hat{\beta}_1\right)$ for a 95% confidence interval

Hypothesis Tests
- Null Hypothesis: $H_0$: No relationship between X and Y or $\beta_1 = 0$ for T tests
	- Multivariate: $H_0: \beta_1=\beta_2=\cdots=\beta_p=0$ for F Tests
- Alternative Hypothesis: $H_1$: There is some relationship between X and Y or $\beta_1 \neq 0$
	- Multivariate: At least one $\beta_j$ is nonzero

T Statistic
- $t=\frac{\hat{\beta}_1-0}{\operatorname{SE}\left(\hat{\beta}_1\right)}$
- Number of standard deviations $\hat{\beta}$ is away from 0
- t distribution is very similar to normal for values greater than 30

P Values
- Small p value indicates that it is unlikely to observe such a substantial association between teh predictor and teh response due to chance, in the absence of any real association between teh predictior and the response
- Small p -> There is a relationship between X and Y. Reject the Null Hypothesis

F Statistic
- $F=\frac{(\mathrm{TSS}-\mathrm{RSS}) / p}{\operatorname{RSS} /(n-p-1)}$
- No relationship between predictors and response expect this to be close to 1. Ideally expect it to be greater than 1
- adjusts for the number of predictors so we won't be skewed by t statistics random chance

