Taken from Probabilistic Machine Learning Chapter 36


**Definitions**
- Spurious Correlation: Causally unrelated variables might happen to be correlated with each other over some period of time

The core idea behind causal models is to create a mechanistic model of the world in which we can reason about the effects of local changes.

Causal Inference: Attempt to answer question where did X cause Y to change. 
- Attempts to determine whether one variable actually drives changes in one another or whether a third factor is driving both. 
- Correlation is not causation: Ice cream production is strongly associated with deaths by drowning vs smoking is strongly associated with lung cancer

Three Steps in Casual Inference:
- Causal estimands: Define quantitates we want to estimate
	- How world would change under intervention
	- The expected number of drownings in the United States if we ban ice cream”
- Identification: Identify estimands stands with quantities that can in principle be estimated from observational data
	- Tells us what questions we could answer with perfect knowledge of data distribution
	- The causal effect is equal to the expected number of drownings after adjusting for month
- Estimation: Estimate the observable quantity using finite data sample.
	- Allows us to exploit non parametric predictive models in ML

**ATE (Average Treatment Effect)**: Estimation of the causal effect of an intervention averaged over all members of the population, know as the average treatment effect or ATE!
- For each unit or person define:
	- $Y(1)$: Outcome if treated
	- $Y(0)$: Outcome if not treated
- $\mathrm{ATE}=E[Y \mid \operatorname{do}(A=1)]-\mathbb{E}[Y \mid \operatorname{do}(A=0)]$
	- On average how much better or worse is getting Treatment then not getting treatment!

## Causal formalism

Goal: the goal is to use data to learn about how the outcome in the world would change under intervention. Need to use our knowledge about causal knowledge of the world. 

## Causal Structural Models

Smoking Example:

$A_i$: indictor on whether or not person $i$ smoked at a particular age. 
$Y_i$: indicator of whether or not person $i$ developed lung cancer at a later age
$H_i$: health consciousness index that measures a person's health consciousness
$G_i$: indicator for whether the person has a genetic predisposition toward cancer

Suppose $\left(A_i, Y_i, H_i\right) \stackrel{\mathrm{iid}}{\sim} p^{\mathrm{obs}}$: Then we can write this using a probabilistic model as follows:
- $A \sim P^{\text {obs }}(A)$
- $H \mid A \sim P^{\text {obs }}(H \mid A)$
- $Y \mid A, H \sim P^{\mathrm{obs}}(Y \mid H, A)$
- $G \mid A, H . Y \sim P^{\text {obs }}(G \mid A, H, Y)$ 

These are all valid probabilistic model however, they are not good causal model. For example, $A$ would be preceded by $H$. Since decision to smoke is probably driven by your health consciousness. 

Casual Structural Models (SCM): 
- Describe a probablistic model
- Provide semantics for transforming the data generating process through intervention. 
- Each variable in a system is determined by combining other modeled variables (the causes) with exogenous “noise” according to some  unknown deterministic function
- $\mathcal{M}=(\mathcal{U}, \mathcal{V}, \mathcal{F})$ 
	- $\mathcal{U}=\left\{U_i: i=1: N\right\}$
		- Set of unexplained or exogenous noise variables passed into model
	- $\mathcal{V}=\left\{V_i: i=1: N\right\}$
		- Endogenous variables that are part of model iself
	- $\mathcal{F}=\left\{f_i: i=1: N\right\}$
		- Set of determinisitc function of the form $V_i=f_i\left(V_{\mathrm{pa}_i}, U_i\right)$
			- $pa_i$ partent of variable $i$ and $U$ are external inputs
- Can set this up in a recursive way, such that depedency graph of nodes given their parents is a DAG. 
- Assume model is causally sufficient: $\mathcal{V},\mathcal{U}$ are all of the causally relevent factors even though they may not be observed at all!
- Anything not modelled are the exogenous terms and use distribution $p(U)$ over their values.

Example for the Smoking one: The SCM for $P^{obs}$ :
$\begin{aligned} & G \leftarrow f_G\left(\xi_0\right) \\ & H \leftarrow f_H\left(\xi_1\right) \\ & A \leftarrow f_A\left(H, \xi_2\right) \\ & Y \leftarrow f_Y\left(G, H, A, \xi_3\right)\end{aligned}$

In this case here
- $f$: fixed unknown functions
- $\xi$: unmeasured causes, modelled as indepdent random noise
- $f_G,f_h,f_A,f_Y$: Deterministic physical relationships in the real world, which $x_i$ are hidden causes that are sufficient to distinguish each $i$ in the population
- $i$ is drawn at random from population therefore modelled as noise

These in turn imply probabilistic models: implies probabilistic model for the observed data based on the factorization:
- $P^{\circ \circ \mathrm{bs}}(G, H, A, Y)= P^{\text {obs }}(G) P^{\text {obs }}(H) P^{\text {obs }}(A \mid H) P^{\text {obs }}(Y \mid A, H)$. 
- Thus, we could sample from the SCM in the same way we would from a probabilistic model: draw a set of noise variables $\xi$ and evaluate each assignment operation in the SCM in order.

SCM encodes additional assumptions about the effects of interventions
- Do-calculus
- Important: $Y=f_Y\left(X, U_Y\right)$
	- Means that $X$ causes $Y$
	- The structure of the equations create different counterfactual worlds and therefore different $ATE$
	- In an SCM, the equations cannot be rearranged without changing the actual causal meaning of the model.
- Interventions represented by replacing assignment statements

Ex: Eliminate Smoking, $A$ , what does this do to Y?
- Second Line would be $A \leftarrow 0$.
- Then probability would become : $P(Y \mid \operatorname{do}(A=0), H)$

### Causal DAG

Casual Graphs are a level of abstraction useful to determining the structure of the problem.

ex: smoking
![[Pasted image 20251201162247.png]]
Left is original setup:
- health consicousness influences smoking decision
- health consiousness influences cancer through lifestyle choice
- Genetic disposition affect cancer
- smoking affects cancer

Right: Assume that Nobody can Smoke so $A=0$

Use Rule: In symbols:
If the original structural equation is:

$$
A=f_A\left(\operatorname{Parents}(A), U_A\right),
$$

then the intervention replaces it with:

$$
A:=0,
$$

and thus:
- Parents(A) are deleted
- Children(A) remain
- If you **do(A=a)**:
	- Remove (cut) all incoming arrows into A.  
		- Erase arrows pointing INTO A
	- Keep all outgoing arrows from A.
		- Do not erase arrows leaving A
	- A is now fixed by an external action, so its old causes no longer determine its value

therefore we get
- $H \to A$ : get deleted but rest stays

#### Conditional Independence

We write $\boldsymbol{x}_A \perp_G \boldsymbol{x}_B \mid \boldsymbol{x}_C$ if $A$ is conditionally independent of $B$ given $C$ in the graph $G$.

Introduce notion of d separation:
4.2.4.1 Global Markov properties (d-separation)

D separation is the rule that tells you whether two sets of variables are conditionally independent in a DAG. If all paths between A and B are blocked then A and B are d separated given C. If any path is open A and B are d-connected , then they are dependent on C. 

We say an undirected path $P$ is $\mathbf{d}$-separated by a set of nodes $C$ (containing the evidence) iff at least one of the following conditions hold: ($C$ is what you condition on and saying that $m$ is basically what you condition on here when we say $m \in C$)
1. P contains a chain or pipe, $s \rightarrow m \rightarrow t$ or $s \leftarrow m \leftarrow t$, where $m \in C$
	1. $s\perp_G t | m$ : Blocked so independent
2. P contains a tent or fork, $s \swarrow^m \searrow t$, where $m \in C$
	1. $s\perp_G t | m$ : Blocked so independent
3. P contains a collider or v-structure, $s \nearrow m \nwarrow t,\quad m \rightarrow d$, where $m$ is not in $C$ and neither is any descendant of $m$.
	1. $s\perp_G t | m$ : FALSE. This is dependent
	2. $s\perp_G t | d$. FALSE. D is a child of m therefore, it is not indepdent. 

Next, we cna say that a set of nodes $A$ is d-separated from a different set of nodes $B$ given a third observed set $C$ iff each undirected path from every node $a \in A$ to every node $b \in B$ is d-separated by $C$. Finally, we define the CI properties of a DAG as follows:

$$
\mathbf{X}_A \perp_G \mathbf{X}_B \mid \mathbf{X}_C \Longleftrightarrow \mathrm{~A} \text { is d-separated from } \mathrm{B} \text { given } \mathrm{C}
$$


This is called the (directed) global Markov property.

Probability and Statistics Facts:
- Conditional Probability:  $P(A|B) = \frac{P(A \cap B)}{P(B)}$
- Bayes Theorem: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
- Conditional Independence: $P(A \cap B) = P(A)P(B)$

Prove mathematically that these are independent. Show that $X \perp Z |Y$
Ex: $X \to Y \to Z$
- $p(x, z  | y) = \frac{p(x,y,z)}{p(y)} = \frac{p(x)p(y|x)p(z|y)}{p(y)} = p(x|y) p(z|y)$
- CP, BT, CI asserted!
Ex: $X \leftarrow Y \to Z$
- $p(x, z  | y) = \frac{p(x,y,z)}{p(y)} = \frac{p(y)p(y|x)p(z|y)}{p(y)} = p(x|y) p(z|y)$
- CP, Cancel terms, CI Asserted!!!
Ex: $X \to Y \leftarrow Z$
- $p(x, z  | y) = \frac{p(x,y,z)}{p(y)} = \frac{p(x)p(z)p(y|x,z)}{p(y)} \neq p(x|y) p(z|y)$
	- The graph does not imply this conditional independence so we cannot reduce this term at all!
	- However X and Z are independent without conditioning

Difference between DAG and PGM?
- PGM encode assumptions about statistical relationships where DAG encode our relationships about causal effects

Causal graphs separate causal vs non-causal reasons for statistical associations
- A is a causal ancestor of D: $A \to ... \to D$
	- This is a directed Path. Changing A changes D
	- $P(D \mid d o(A=a)) \neq P\left(D \mid d o\left(A=a^{\prime}\right)\right)$
- A and D share a common ancestor: $A \leftarrow C \to D$
	- A and D are statisticall associated
		- $P(D \mid A=a) \neq P\left(D \mid A=a^{\prime}\right)$
		- Statistically associated because they share a common cause
	- But Interventions on A do NOT affect D:
		- $P(D \mid d o(A=a))=P\left(D \mid d o\left(A=a^{\prime}\right)\right)$
		- Non causal assoication
	- Backdoors mean that we can created statistical association without causation!


##### Graph Surgery

- $A=a$ means delete all incoming edges into A and condition on A=a in resulting probability distribution. 
- $\mathrm{P}(\mathbf{X} \mid \operatorname{do}(A=a))$
	- Distribution of X given A=a under mutilated graph
	- $\mathbb{E}[\mathbf{X} \mid \operatorname{do}(A=a)] \triangleq \mathbb{E}_{\mathrm{P}(\mathbf{X} \mid \operatorname{do}(A=a))}[\mathbf{X}]$

**ATE (Average Treatment Effect)** : $\mathrm{ATE}=E[Y \mid \operatorname{do}(A=1)]-\mathbb{E}[Y \mid \operatorname{do}(A=0)]$
- First Term is the Causal Estimand
	- Can't learn this from the data
		- can't tell where treatment had strong effect vs treatment had no effect do to some unoberserve common binary cause $U$
		- Treatment and outcome are affected by $U$
			- When $U=1$ this increases the treatment and probability of positive outcome


Example of $Y,H,A,G$
- Without Intervention: 
	- $\mathrm{P}(Y, H, G \mid A=a) = \frac{P(Y,H,G,A=a)}{P(A)} = \frac{P(H)P(G)P(A|H)P(Y|A,H,G)}{P(A)} = P(H|A)P(G)P(Y| A=a,G,H)$
- With Intervention:
	- $\mathrm{P}(Y, H, G \mid do(A=a)) = \frac{P(Y,H,G,A=a)}{P(A=a)} = \frac{P(H)P(G)P(A)P(Y|A=a,H,G)}{P(A=a)} = P(H)P(G)P(Y| A=a,G,H)$
- Difference is that $H$ without intervention is observed among smokers while with intervention it is with all individuals.
	- $P(A|H=smoker)$ would lead to more mass on lower values of $H$ than $P(H)$ 
	- Includes hypothesis of how smoking would affect the subpopulation of individuals who tend to bee to health conscious to smoke in the observed data

#### Identification

Observed data gets you correlations. Need to make assumptions about how the world works. They can range from the SCM equations, 

May different SCM can produce the same distributions! So unique data does not mean that we can determine the SCM it came from!

Augment observed with knowledge about the underlying causal structure of the process under consideration. This narrows space such that only one value of causal estimand that is compatible with the observed data

Causal estimand is the considered identified. If causal estimand can be written entirely in terms of observable probability distributions then it is identified. Such a function of observed distributions is a statistical estimand.
- If given data+assumptions there is only one possible value that the causal effect can take then a causual estimand is identified
- i.e in terms of distributions without do operators then done!

An exampple of Statistical estimatnd in SCM above shows that following is true:

$$
\mathrm{ATE} \stackrel{(*)}{=} \tau^{\mathrm{ATE}} \triangleq \mathbb{E}[\mathbb{E}[Y \mid H, A=1]-\mathbb{E}[Y \mid H, A=0]]
$$
- This holds due to some specific properties of the SCM
- There are no do operators here
- so $\tau^{ATE}$ is a function of observable probability distributions!
	- Right hand side is just
	- average cancer rate of smokers of H
	- average cancer rate among non smokers of type H

What assumptions can we make about SCM to conclude this?
- Parametric Assumptions: probability of cancer is additive on logit scale
- Monotonicity Assumption: shape constraint
	- smoking never decreases probability of cancer
- no direct effect
	- smokign is influenced by H not by G
	- causal structural assumption!

Identification requires _some_ assumptions; the game is to choose the **weakest plausible** set that still yields an identified causal effect.
- If we are not willing to make any assumptions about the functional form of the SCM, then our assumptions are just about which variables affect (and do not affect) the other variables.
	- Maybe be incomplete since we won't know whether each possible edge is present in physical system
This follows directly from the ill-posedness of causal problems: if the assumptions used to identify causal quantities could be validated, that would imply that the causal estimand was identifiable from the observed data alone.

##### Causal hierachy

Observational Queries: questions that are purely about statistical associations
- Are smoking and lung cancer associated
Interventional Queries: questions about causal relationships at population level
- how much does smoking increase the probability of lung cancer in a given population
- Concerned with intervention to change outcome. If i stop somebody from smoking what is probability they get lung cancer
	- About noise variables $\xi$ in SCM.
	- unmeasured factors speific to certain individuals
	- Interventional queries are about populations using do notation $P(y | do(A=0))$
		- distribution of lung cancer outcomes for individual who was prevented from smoking
- Require a causal graph and standard assumptions
Counterfactual queries: how an overved outcome may have been different had an intervention been applied in the past. 
- Did most smokers with lung cancer develop cancer because they smoked
- require comparison of counterfactual outcomes within individuals
- Run same values of $\xi$ through different outcomes!
- Written in terms of potential outcomes notation!
	- $Y_i(a) \triangleq f_Y\left(G_i, H_i, a, \xi_{3, i}\right)$
	- Outcome we would have seen had A been set to $a$ while rest held constant
- Require 
	- knowledge about functional forms of all structural equations
	- joint distribution of noise terms
	- cross wrold relationships for same person

ATE Forms: 
- canonical intervention query: $ATE= E[Y_i(1) - Y_i(0)]$
- Need to ask does the query require information about joint distribution of outcomes or does marginal suffice?
- Counterfactual requires conditoning on one value of one outcome 
	- Lung cancer rate among smokers who developed cancer, had they not smoked
	- $\mathbb{E}\left[Y_i(0) \mid Y_i(1)=1, A_i=1\right]$
	- Cant be written using do notation since it requires distinction between y0 and y1
	- Need assumptions about functional forms of the distributiion
	- Did smokers who developed cancer have lower non-smoking cancer risk than smokers who did not develop cancer?
		- requires assumptions about characterstics encoded in $\xi$ translated to cancer outcomes!
	- Need indivudal level data here!


ATE is an interventional estimand (not counterfactual)
Even though ATE is often written using potential outcomes:
$$
A T E=\mathbb{E}[Y(1)-Y(0)]
$$
it is equivalent to:
$$
A T E=\mathbb{E}[Y \mid \operatorname{do}(A=1)]-\mathbb{E}[Y \mid \operatorname{do}(A=0)]
$$
and can be identified from observed data under typical assumptions.
ATE does not require individual-level counterfactual reasoning.


##### Randomized Control Trials

Randomized Control Trials
- experiments designed to make the causal concerns as simple as possible
- Assign each person to treatment or control at random
	- no causes of treatment assignment therefore no common causes of $Y$ and $A$
		- No variable causes A anymore. It is completely a random variable!
		- IN the GRAPh nothing goes into A anymore!!!!
		- If no parents in RCT, the random assignment which means the below holds true 
	- $\mathrm{P}(Y \mid \operatorname{do}(A=a))=\mathrm{P}(Y \mid a)$
		- A has no parents, the mutilated graph is therefore the same as the original graph!
- USE RCT to study the ATE:
	- $\mathrm{ATE}=E[Y \mid \operatorname{do}(A=1)]-\mathbb{E}[Y \mid \operatorname{do}(A=0)]$: Theoretical Causal Estimate
	- Expected differenced between being assigned treatment and assigned no treatment for a randomly chosen member of the population
	- $\tau$ is the treatment effect parameter, a number that measures the causal effect of treatment A on outcome Y. 
		- Identified ATE/Estimate version of ATE based on assumptions and causal model
		- $\tau^{A T E}=\mathbb{E}[\mathbb{E}[Y \mid H, A=1]-\mathbb{E}[Y \mid H, A=0]]$
	- IN RCT setting we get a very simple $\tau$
		- $\tau^{\mathrm{RCT}}=\mathbb{E}[Y \mid A=1]-\mathbb{E}[Y \mid A=0]$
		- Estimator: $\hat{\tau}^{\mathrm{RCT}} \triangleq \frac{1}{n_{\mathbf{A}}} \sum_{i: A_i=1} Y_i-\frac{1}{n-n_{\mathbf{A}}} \sum_{i: A_i=0} Y_i$
			- number of treated units
			- number of controls
			- average outcome of treated - avereaged outcome of untreated
		- Why does the do operator get removed in the RCT setting?
Given randomization:
$$
A T E=\mathbb{E}[Y \mid \operatorname{do}(A=1)]-\mathbb{E}[Y \mid \operatorname{do}(A=0)]
$$
But in an RCT:
$$
\mathbb{E}[Y \mid d o(A=a)]=\mathbb{E}[Y \mid A=a]
$$
So:
$$
A T E=\mathbb{E}[Y \mid A=1]-\mathbb{E}[Y \mid A=0]=\tau^{R C T}
$$
Hence the do-operator disappears.


RCT are gold standard for estimating causal affections. By design there are no cofounders that can prduce alternative causal explanations. 
- **Cofunders**: variable that influences both $A$ and $Y$ 
	- $H \to A \to Y$
	- $H \to Y$
	- H is a confounder since it sends arrows to both!

A confounder is a variables C that lies on a backdoor path between A and Y: 
$A \leftarrow C \to Y$.

This creates a non-causal association between A and Y.
- Need to block or adjust for C or else you mix up the
	- causal affect of $A \to Y$
	- spurious association of $C \to A$ and $C to Y$
	- 

Additionally, it may be difficult to ensure that the participants in an RCT are representative of the population where the treatment will be deployed. For instance, participants in drug trials may skew younger and poorer than the population of patients who will ultimately take the drug

##### Confounder Adjustment

ATE is also called ACE (Average Causal Effect)

Situation where $A$ is binary, observe outcome $Y$ as well as a set of common causes $X$ that influence both $A$ and $Y$

###### Causal estimand, statistical estimand, and identificatio

Treatment: $A$
Outcome: $Y$
Covariates: $X$

are draw iid from $P$. 

Goal: Determine the expected difference between being assigned treatment and assigned no treatment for a randomly chosen member of the population. 
Three Steps
1) Mathematically formalize the causal estimand
2) Give conditions for the causal estimand to be identified as a statistical estimand
3) Finally estimate the statistical estimand from data

Step 1)
- $\mathrm{ATE}=\mathbb{E}[Y \mid \operatorname{do}(A=1)]-\mathbb{E}[Y \mid \operatorname{do}(A=0)]$

Step 2)
- Articulate sufficient conditions for the ATE to be identified as a statistical estimand.
	- Issue is presence of confounders.
	- Confounders: common cause variables that affect both the treatment and outcome
	- When confounders exist, the sub-population of people who are observed to have received one level of the treatment A will differ from the rest of the population in ways that are relevant to their observed Y .
		- EX: Horseback riding (treatment) and healthiness (outcome) have a positive association. However, both influenced by confound variable wealth $X$
		- The population of people who rode horses as children (A =1) is wealthier than the population of people who did not. Accordingly, the horseback-riding population will have better health outcomes even if there is no actual causal benefit of horseback riding for adult health.
- Say we have the following $DAG$ $X \to A$, $A \to Y$, $X \to Y$
	- In identification for this ATE remove Edge where $X \to A$ (eliminate parents according to do! that is what it means -> ) **$A \rightarrow Y \leftarrow X$**
	- The strucutral causal equation that generates Y is the same still!
		- The **mechanism for Y** stays the same
		- The **distribution of X** stays the same
		- But the distribution of A does not stay the same 
	- Original Situation
		- $\begin{aligned} X & =f_X\left(\varepsilon_X\right) \\ A & =f_A\left(X, \varepsilon_A\right) \\ Y & =f_Y\left(A, X, \varepsilon_Y\right)\end{aligned}$
		- $p(X, A, Y)=p(X) p(A \mid X) p(Y \mid A, X)$
	- New Sitatuon
		- $\begin{aligned} X & =f_X\left(\varepsilon_X\right) \\ A & =a \quad(\text { constant; no dependence on } X) \\ Y & =f_Y\left(a, X, \varepsilon_Y\right)\end{aligned}$
		- $p(A \mid d o(A=a))=\delta(A=a)$, $p(X, Y \mid d o(A=a))=p(X) p(Y \mid A=a, X)$
Theorem 2 (Adjustment with no unobserved confounders). We observe $A, Y, X \sim P$. Suppose that
1. (Confounders observed) The data obeys the causal structure in Figure 36.3. In particular, $X$ contains all common causes of $A$ and $Y$ and no variable in $X$ is caused by $A$ or $Y$.
	1. After conditioning on X, treatment and outcome are uncofounded
2. (Overlap) $0<\mathrm{P}(A=1 \mid X=x)<1$ for all values of $x$. That is, there are no individuals for whom treatment is always or never assigned.
	1. Ensures that every $X$ value has both treatest and untreated people
	2. This allows use to estimate conditional expectations like $E[Y \mid A=a, X=x]$
	3. Definition 1. A distribution $P$ on $A, X$ satisfies overlap if $0<P(A=1 \mid x)<1$ for all $x$. It satisfies strict overlap if $\epsilon<\mathrm{P}(A=1 \mid x)<1-\epsilon$ for all $x$ and some $\epsilon>0$.

Then, the average treatment effect is identified as $\mathrm{ATE}=\tau$ (staitistical estimand), where

$$
\tau=\mathbb{E}[\mathbb{E}[Y \mid A=1, X]]-\mathbb{E}[\mathbb{E}[Y \mid A=0, X]] .
$$
- **Backdoor Adjustment Formula**

Proof. First, we expand the ATE using the tower property of expectation, conditioning on $X$. Then, we apply the invariances discussed above:

$$
\begin{aligned}
A T E & =\mathbb{E}[Y \mid \operatorname{do}(A=1)]-\mathbb{E}[Y \mid \operatorname{do}(A=0)] \\
& =\mathbb{E}[\mathbb{E}[Y \mid \operatorname{do}(A=1), X]]-\mathbb{E}[\mathbb{E}[Y \mid \operatorname{do}(A=0), X]] \\
& =\mathbb{E}[\mathbb{E}[Y \mid A=1, X]]-\mathbb{E}[\mathbb{E}[Y \mid A=0, X]]
\end{aligned}
$$
-  First Equation is ATE
- Second is Tower Property: $E[Y] = E[E[Y|X]]$
- third is under intervention: The expected outcome **if** A were _externally forced_ to be a,  
no matter how A would normally be determined. The do means $A=a$

How does $\tau$ differ from naive estimate: $\mathbb{E}[Y \mid A=1]-\mathbb{E}[Y \mid A=0]$
- The naive does not account for confounding

$$
\begin{aligned}
&\tau=\int \mathbb{E}[Y \mid A=1, X] P(X) d X-\int \mathbb{E}[Y \mid A=0, X] P(X) d X\\
&\text { We can write the naive estimand in a similar form by applying the tower property of expectation: }\\
&\mathbb{E}[Y \mid A=1]-\mathbb{E}[Y \mid A=0]=\int \mathbb{E}[Y \mid A=1, X] P(X \mid A=1) d X-\int \mathbb{E}[Y \mid A=0, X] P(X \mid A=0) d X
\end{aligned}
$$
- - The **causal ATE** averages over the **same** distribution $P(X)$
	- For each value of X, compare treated vs untreated **within the same X**, then weight by the **true population distribution** of X.
- The **naive association** averages over **different** distributions $P(X∣A=1)P(X \mid A=1)$ and $P(X∣A=0)P(X \mid A=0)$
	- “Look at treated people and average over **their** X distribution.  Look at untreated people and average over **their** X distribution.”


##### ATE Estimation with observed confounders

Estimating ATE using finite data sample. Known as backdoor adjustment. Recall that the defining characteristic of a confounding variable is that it affects both treatment and outcome.

##### Outcome Model Adjustment

We begin with an approach to covariate adjustment that relies on modeling the conditional expectation of the outcome Y given treatment A and confounders X. Outcome model adjustment relies on modeling the relationship between the confounders and the outcome.

Outcome Adjustment,
Define the conditional expected outcome as  $Q$ as : $Q(a, x)=\mathbb{E}[Y \mid A=a, X=x]$

Substitute into $\tau$: $$\tau=\mathbb{E}[\mathbb{E}[Y \mid A=1, X]]-\mathbb{E}[\mathbb{E}[Y \mid A=0, X]] .$$
$$
\tau=\mathbb{E}[Q(1, x)-Q(0, x)]
$$
Therefore we can fit a model $\hat{Q}$ for $Q$ then report: 
$$
\hat{\tau}^Q \triangleq \frac{1}{n} \sum_i \hat{Q}\left(1, x_i\right)-\hat{Q}\left(0, x_i\right)
$$
Recall that $$
E[Y \mid a, x]=\operatorname{argmin}_Q \mathbb{E}\left[\left(Y-Q(A, X)^2\right]\right.
$$
So we use a mean squared error to fit a predictor that predicts $Y$ from $A$ and $X$

So steps:
1) Fit a model $\hat{Q}$ to predict Y
2) For each unit $i$ predict that unit's outcome had they received treatment $\hat{Q}(1,x_i)$ and predict outcome had they not received treatment $\hat{Q}(0,x_i)$
3) If unit did received treatment then $\hat{Q}(0,x_i)$ is our guess for if they did not!
4) Estimated gain would be $\hat{Q}\left(1, x_i\right)-\hat{Q}\left(0, x_i\right)$
	1) The difference in expected outcome between being treated and not treated
5) Estimate our expected with respect to $P(X)$- true population distribution of confounders
	1) $\hat{\mathrm{P}}(X)=1 / n \sum_i \delta_x$
	2) Substitute the expectation (over an unknown distribution) by an average over the observed data

What happens when using Linear Regression?
- $Q(A, X)=\mathbb{E}[Y \mid A, X]=\mathbb{E}\left[f_Y(A, X, \xi) \mid A, X\right]=\beta_0+\beta_A A+\beta_X X$
- $Q(1, X)-Q(0, X)=\beta_A\left(\right.$ and so also $\left.\tau=\beta_A\right)$
- So estimator reduces to the estimator for the regression coefficient $\beta_A$
	- This is a common way to estimate the association between two variables in practice

Example:
Age (A), Treatment (X) and blood pressure (Y)
- $\hat{Q}(a, x)=\hat{\beta}_0+\hat{\beta}_A a+\hat{\beta}_X x$
- say calculated was $\hat{Q}(a, x)=110+10 a+1 x$
- Plug in for each xi
	- Outcome if treated: $\hat{Q}\left(1, x_i\right)=110+10+x_i$
	 - Outcome if untreated: $\hat{Q}\left(0, x_i\right)=110+0+x_i$
 - compute $\hat{\tau}=\frac{1}{6} \sum_{i=1}^6\left(\hat{Q}\left(1, x_i\right)-\hat{Q}\left(0, x_i\right)\right)$
 - 



Summary and practical advice

We have seen a number of estimators that follow the general procedure:
1. Fit statistical or machine-learning models $\hat{Q}(a, x)$ as a predictor for $Y$, and/or $\hat{g}(x)$ as a predictor for $A$
2. Compute the predictions $\hat{Q}\left(0, x_i\right), \hat{Q}\left(1, x_i\right), \hat{g}\left(x_i\right)$ for each datapoint, and
3. Combine these predictions into an estimate of the average treatment effect.