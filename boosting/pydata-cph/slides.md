---
title: "Gradient Boosting:<br>How does it work?"
author: Christian Duffau-Rasmussen
date: 9. November 2021
---

### 

:::::::::::::: {.columns}
::: {.column width="50%"}

[![](../static/ante_logo_neg.svg){height=20%}](https://ante.dk)

:::
::: {.column width="50%"}
- Data Scientist and Co-founder

- Legal tech start-up

- NLP-powered search engine

- Extract meta data from plain text

- Index structured data in a search engine 
:::
::::::::::::::


## Boosting - What is it ?


> A procedure to combine many __weak__ learners to produce a powerful __committee__. 
>
> Elements of Statistical Learning [@friedman2009elements, sec. 10.1]

::: notes

- An ensemble method in Machine Learning
- A way of combining so-called weak learner
- into a powerful comittee i.e. by some means og aggregation or averaging

:::

## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](../timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](../static/leslie-valiant.jpg){height=130px}

[@valiant1984theory]

- Probably Approximately Correct Learner (PAC)  
- A formal theory of learnability 
- Proves broad classes of non-trivial boolean functions are learnable
- Turing Award in 2010

:::
::::::::::::::

::: notes

- Leslie Valiant - Harvard
- Leslie Valiant (1984) - A theory of the learnable
- Defines a mathematical framework for analyzing what classes of problems are learnable in polynomial time. 
- Introduces the Probably Approximately Correct Learner (PAC-learner). 
- Foundation of the field of computational learning theory.
- The PAC leaner is to
  - computational learning theory what
- The Turing machine is to
  - computational complexity theory 
:::

### Side note: PAC learning

- Boolean _concept_ : $(0,1,1,0,\cdots, 1) \mapsto 0 \text{ or } 1$

- A _concept_ is Probably Approximately Learnable
  -  _if_ an _algorithm_ can deduce a _hypothesis_ (aka. a function)
  - in time bounded by a polynomial of the size of _concept_ in bits
  - with $$P\left(\text{error-rate} <\varepsilon\right) > \delta$$
  - _for all_ $\varepsilon>0$ and $0 < \delta \leq 1$

- The _learning protocol_ is:
  -  learn from examples asking an ORACLE
  -  the ORACLE returns a random example in _unit_ time (O(1))

## A brief history


:::::::::::::: {.columns}
::: {.column width="66%"}
![](../timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![Leslie Valiant](../static/leslie-valiant.jpg){height=130px}
![Michael Kearns](../static/michael-kearns.jpg){height=130px}

[@kearns1989crytographic]

- Introduce _weak learner_  
- Performs only slightly better than chance
- _Hypothesis boosting problem_: weak <=> strong ?

:::
::::::::::::::

::: notes
- Michael Kearns & Leslie Valiant (ph.d. student and supervisor) - Harvard University
- The 1989 Crypto paper show that
- If either: 
  - Boolean formulae
  - Deterministic finite automata 
  - Constant-depth threshold circuits 
- are learnable then cryptography is toast.

- Kearns and Valiant state as an open problem:
  - Can weak learners be "boosted" into strong learners?
  - I.e. can an algorithm transform weak learners in to strong ones
  - The notion at the time was "probably not" 

:::

### Side note: _weak_ and _strong_ learners

- _strongly learnable_ == PAC learnable

- A _concept_ is _weakly_ Learnable
  -  _if_ an _algorithm_ can deduce a _hypothesis_
  - in time bounded by a polynomial of the size of _concept_ in bits ($s$)
  - with $$P\left(\text{error-rate} < \frac{1}{2} - 1/p(s)\right) > \delta$$
  - _for all_ $0 < \delta \leq 1$

## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](../timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](../static/robert-schapire.jpg){height=130px}

[@schapire1990strength]

- Cracks the _Hypothesis boosting problem_
- Shows _weak learner_ <=> _strong learner_ 
- An algorithm constructing a strong learner from weak ones 🤯 

:::
::::::::::::::

::: notes
- Robert E Schapire - MIT,later professor at Princeton - 27 years at the time
- Ph.d. thesis
:::

## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](../timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](../static/yoav-freund.png){height=130px}

[@freund1990majority]

- Youav Freund - Israeli - p.hd. - UC Santa Cruz - 29 years at the time
- Ph.d. thesis
- Implements a much more efficient _boosting_ algorithm
- Trains learners on weighted subsets of the data 
- Uses majority voting to predict

:::
::::::::::::::

## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](../timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](../static/yoav-freund.png){height=130px}
![](../static/robert-schapire.jpg){height=130px}

[@schapire1995decision]

- Youav Freund - Israeli - Post Doc at UC San Diego - 34 years at the time
- Robert E Schapire - Post doc at Princeton - 32 years at the time
- Introduces AdaBoost
- First practical boosting algorithm  
- Winners of Gödel prize in 2003

:::
::::::::::::::

::: notes
- Motivate AdaBoost by an Hedging example
- Choose the allocation of money to gamblers placing bets on your behalf
- Minimize difference to best performing gambler in a online scheme
- Translate the hedging algo to a training setup
- Derive bounds on errors of the Adaboost outputted hypothesis
:::

## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](../timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](../static/jerome-h-friedman.jpeg){height=130px}
![](../static/robert-tibshirani-trevor-hastie.jpg){height=130px}

[@friedman2000special]

- Shows AdaBoost is *Stagewise Additive Logistic regression*

:::
::::::::::::::

::: notes
- Jerome Harold Friedman - American - Professor of Statistics at Stanford University - 61 years at the time
- Trevor Hastie - South Africa - Professor in Statistics at Stanford University - 47 years at the time
- Robert Tibshirani - Canadian - Professor in Statistics at Stanford University - 44 years at the time

:::

## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](../timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](../static/jerome-h-friedman.jpeg){height=130px}

[@friedman2001greedy]
[@mason1999boosting]

- Generalizes the _boosting_ concept 
- Describes _boosting_ as gradient descent in function space  

:::
::::::::::::::

## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](../timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](../static/tianqi-chen.jpg){height=130px}

[@chen2015higgs]

- Wins Kaggle contest on Higgs Boson using XGBoost 
- XGBoost quickly becomes the most winning algorithm  

:::
::::::::::::::

::: notes

- Tianqi Chen - Shanghai China - Ph. d. at Univeristy of Washington
:::

## Decision trees

:::::::::::::: {.columns}
::: {.column width="66%"}
![Top left: General partition, Top right: recursive binary splits](../static/partitions-and-trees.png){width=80%}
:::

::: {.column width="33%"}
- Partitions feature space into rectangles
- Finding Optimal trees is NP-complete
- Greedy algorithms and heuristics are used when fitting (CART, ID3, C4.5, etc.)
- Trees are fast when predicting
:::
::::::::::::::


::: notes
- Scikit-learn uses an optimized version of CART
- Heuristics are used to approximate a optimal solution
- Most fitting procedures uses some kind of recursive binary splitting
:::

## Ensemble methods

- Bagging: Grow trees using _random subsets of data (with replacement)_
- Random forest: Grow trees using _random subset of features_
- Boosting: Grow trees on _re-weighted dataset_

. . . 

$$\text{Boosting} \succ \text{Random forest} \succ \text{Bagging} \succ \text{Tree}$$

::: notes
- Ensemble methods overcome overfitting by combining trees (typically)
:::

## Why Ensembles of Decision trees?

:::::::::::::: {.columns}
::: {.column width="66%"}
![](../plots/tree-depth.svg){width=90%}
:::

::: {.column width="33%"}
- Nonlinearity
- Feature interaction:
  - 1 layer: $f(X_i)$ 
  - 2 layers: $f(X_i, X_j)$
  - 3 layers: $f(X_i, X_j, X_k)$ 
- Automatic feature selection
- Single Trees overfit 
- Combining many small trees ...
- .. you get *flexible* yet *robust* models 
:::
::::::::::::::

::: notes
- Ensemble methods most often use trees
- But why?
- Shallow Trees are weak learners, so don't overfit 
- Shallow Trees still model interactions
- Linear model cannot model interactions
- Automatic feature selection
:::


## Simulation example

$$Y = \begin{cases}
1 & \text{if}\quad X_1^2 + \ldots + X_{10}^2 > 9.34 \\
-1 & \text{else}
\end{cases}\quad X_i\sim N(0,1)$$

![Simulated 10-D nested spheres](simulation/plots/gen_data.png){height=400px}

## Simulation example

![](simulation/plots/ensemble_test_errors.svg)


::: notes
- Adaboost outperforms the other ensemble methods with stumps vs full trees
:::
## Simulation example

![](simulation/plots/adaboost_errors.svg)

::: notes
- Adaboost test error decreases after training error has hit _zero_
- Adaboost learns something more general than just the training data "by heart"
:::

## Bias/variance trade-off

$$\text{MSE} = \text{Bias}(\hat{f})^2 + \text{Var}(\hat{f}) + \text{Irreducible noise}$$

## Variance of averages

$$\text{Var}\left(\frac{1}{n}\sum_i^n X_i\right) = \frac{1}{n}\sum_i^n \text{Var}\left(X_i\right) + \frac{1}{n}\sum_{i\neq j} \text{Cov}(X_i, X_j)$$

. . .

$$\text{Variance of ensemble} = \frac{\text{Var(Trees)}}{n} + \frac{\text{Cov( Trees)}}{n}$$

## Variance and bias reduction

![Classifying 10-D nested spheres](simulation/plots/consecutive_predictions_corr.svg)

::: notes
- Bagging: Training data subsets correlates. Mellow positive correlation between all models.
- Random forest: 
  - Either correlation of 1: Set of features with equal importance
  - Correlation of zero: Features which are uncorrelated
- Adaboost: Zero or negative correlation. Out of thin air.
:::

## Boosting

![](../static/Ensemble_Boosting.svg)

::: notes

- A collection of _weak learners_ (e.g. classifier) are trained sequentially.
- Each _learner_ is trained on the same dataset.
- Each example is re-weighted in each iteration
- Poorly predicted examples get higher weight 
- Well predicted examples get lower weight
:::

## Adaboost algorithm

```python
def fit(X, y, weak_leaner, iters=100):
  weights = np.ones(len(y)) * 1/len(y)
  for i in range(iters):
      weak_leaner[i].fit(X, y, sample_weight=w)
      y_pred = weak_leaner[i].predict(X)
      errors = (y_pred != y)
      error_rate = sum(weights * errors)/sum(w)
      alphas[i] = log((1 - error_rate)/error_rate)
      weights = weights * errors * exp(alpha)
```

1. Initialize weights $w_i=1/N$
2. For $m=1$ to $M$:
   1. Fit classifier $G_m(x)$ to training data using $w_i$'s
   2. Compute $$\text{err} = \frac{\sum_{i=1}^N w_i I(y_i \neq G_m(x_i))}{\sum_{i=1}^N w_i}$$
   3. Compute $\alpha_m = \log((1-\text{err})/\text{err})$ 
   4. Set $w_i \leftarrow w_i \exp(\alpha_m I(y_i \neq G_m(x_i)))$
3. Output $$G(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m G_m(x)\right)$$


## Adaboost

- Not originally motivated in _Forward Stagewise Additive Learning_
- [@friedman2000special] show that Adaboost is a stagewise additive model with loss $$L(y, f (x)) = \exp(−y f(x))$$ where $y\in \{-1, 1\}$ and $f(x) \in \mathbb{R}$.
- Usually for classification we use cross-entropy 
\begin{align}
L(y, f(x)) &= y^\prime \log p(x) + (1-y^\prime) \log(1- p(x)) \\ 
&= \log(1 + e^{-2yf(x)})
\end{align} 
where $y^\prime = (y+1)/2\in \{0,1\}$ and $p(x)$ is the softmax function.

### Cross-entropy derivation

\begin{align}
P(y=1|x) &= p(x) = \frac{e^{f(x)}}{e^{-f(x)}+e^{f(x)}} = \frac{1}{1+e^{-2f(x)}} \\
P(y=-1|x) &= 1-p(x) = \frac{e^{-f(x)}}{e^{-f(x)}+e^{f(x)}} = \frac{1}{1+e^{2f(x)}}
\end{align}

\begin{align}
L(y, f(x)) &= \log p(x) =  \log \frac{1}{1+e^{-2f(x)}} \quad\text{if}\,y=1 \\
L(y, f(x)) &= \log(1- p(x)) = \log \frac{1}{1+e^{2f(x)}} \quad\text{if}\,y=-1  \\ 
\end{align}
hence the _negative log-likelihood_ is the _cross-entropy_
\begin{align}
L(y, f(x)) &= \log \frac{1}{1+e^{-2yf(x)}}  \Leftrightarrow \\
-L(y, f(x)) &= \log \left(1+e^{-2yf(x)}\right)
\end{align}

## Adaboost

- In theory the two loss functions are equivalent
- On average they produce the same _model_ $f$
- For finite samples the exponential loss has drawbacks
- To much weight is given to errors

. . .

![Exponential loss and cross-entropy](../plots/loss_functions.svg){width=50%}

## Adaboost

- 10D nested spheres with noisy data

:::::::::::::: {.columns}
::: {.column width="50%"}
![Clean data](simulation/plots/gen_data.png)
:::
::: {.column width="50%"}
![Noisy data](simulation/plots/gen_noisy_data.png)
:::
::::::::::::::


## Adaboost

- Adaboost has trouble on noisy data

. . .

:::::::::::::: {.columns}
::: {.column width="50%"}
![Test error clean data](simulation/plots/ensemble_test_errors.svg)
:::
::: {.column width="50%"}
![Test error noisy data](simulation/plots/ensemble_test_errors_noisy.svg)
:::
::::::::::::::


:::notes
- The deviance yf is only negative if they have different signs
- So all negative values are errors
- The exponential loss eight these error much heavier
:::

## Can we fix Adaboost ?

- Can we easily plug in a more robust loss function? 

. . .

> ::: nonincremental
> 1. Initialize weights $w_i=1/N$
> 2. For $m=1$ to $M$:
>    1. Fit classifier $G_m(x)$ to training data using $w_i$'s
>    2. Compute $\text{err} = \frac{\sum_{i=1}^N w_i I(y_i \neq G_m(x_i))}{\sum_{i=1}^N w_i}$
>    3. Compute $\alpha_m = \log((1-\text{err})/\text{err})$ 
>    4. Set $w_i \leftarrow w_i \exp(\alpha_m I(y_i \neq G_m(x_i)))$
> 3. Output $$G(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m G_m(x)\right)$$
> :::

. . .

🤷‍♂️

## Forward Stagewise Additive Learning


```python
def estimator(X):
  pass
```
### Forward Stagewise Additive Learning

1. Initialize $f_0(x) = 0$
2. For $m=1$ to $M$:
   a. Compute $$(\beta_m, \gamma_m) = \underset{\beta, \gamma}{\text{argmin}} \sum_{i=1}^N L(y_i, f_{m-1}(x_i)+ \beta b(x_i;\gamma))$$
   b. Set $f_m(x) = f_{m-1}(x) + \beta_m b(x; \gamma_m)$

::: notes

- Boosting is a special case of _Forward Stagewise Additive Learning_
- It's an approximation technique for general additive models

:::


## Gradient boosting 

- Introduced by [@friedman2001greedy] as a new view of boosting
- Makes it possible to _derive_ a boosting procedure
- Solves each forward stagewise step as _gradient descent_ in a function space
- Applies to any differentiable loss function


## Side note: Gradient boosting in SciKit Learn

_Scikit-learn documentation_:

> GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.

. . .

Sounds promising...

. . .

> `loss: {‘deviance’, ‘exponential’}`
>
> For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.

. . .

🤦‍♂️

## Gradient boosting

- Consider the loss function,
$$\sum_{i=1}^N L\left(y_i, \sum_{m=1}^M \beta_m b(x_i, \gamma_m)\right)$$ 
- ... to be optimized over $\{\beta_m, \gamma_m\}_{m=1}^M$
- ... which might me intractable

## Gradient boosting
- We try to solve it in a Forward stagewise mamner
- For $1$ to $M$:
  - Optimize $$\sum_{i=1}^N L\left(y_i, f_{m-1}(x_i) + \beta_m b(x_i, \gamma_m)\right)$$ 
- where $f_m(x) = f_{m-1}(x) + \beta_m b(x_i, \gamma_m)$.
- If the basis functions are trees
- ... the stagewise minimization ca be difficult 

## Gradient boosting

- Consider the expected loss function $$\phi(f) = E[L(y,f(\boldsymbol{x}))]$$ 
- we seek $f$ that minimizes $\phi$
- Thats a difficult variational analysis problem
- Let's circumvent the problem... 😎
- ... consider a dataset of $N$ observations instead $$\mathbf{f} = \{f(x_1), \ldots, f(x_N)\}$$


## Gradient boosting

- Lets minimize the empirical loss $$L(f) = \sum_{i=1}^N L(y_i, f(x_i))$$ wrt. $f(x_i)$ numerically.
- All numerical optimization is a series of $M$ steps towards optimum 
- Let each step be a vector $\mathbf{h}_m\in\mathbb{R}^N$
- The numerical optimum is $$\mathbf{f}_M = \sum_{m=1}^M \mathbf{h}_m$$

## Gradient boosting
- In _gradient descent_ each step $\mathbf{h}_m$ is $$-\rho \mathbf{g}_m$$ 
- ... where $g_{im}$ is $$\left[\frac{\partial L(y_i, f(x_i))}{\partial  f(x_i)}\right]_{f(x)=f_{m-1}(x)}$$
- The numerical optimization is then 
$$
\begin{align*}
  \mathbf{f}_m &= \mathbf{f}_{m-1} + \mathbf{h}_m\\
  &= \mathbf{f}_{m-1} -\rho \mathbf{g}_m 
\end{align*}
$$

## Gradient boosting

- Let's add _line search_, $$\mathbf{f}_m = \mathbf{f}_{m-1} - \rho_m \mathbf{g}_m$$ 
  where $\rho_m = \underset{\rho}{\text{argmin}}\quad L(\mathbf{f}_{m-1}-\rho \mathbf{g}_m )$
- This fits the pattern of _forward stagewise learning_ $$f_m(x) = f_{m-1}(x) + \beta_m b(x; \gamma_m)$$
- where $\rho_m$ is analogous to $\beta_m$
- and $-\mathbf{g}_m$ is analogous to $b(x; \gamma_m)$ 
- We can derive the gradient considering $f(x)$ a variable
- ... and easily compute $\mathbf{g}_m(f(x))$ using the value from the previous iteration $f_{m-1}(x)$
- ... we cannot compute $\mathbf{g}_m$ outside the dataset 🤔

## Gradient boosting

- The idea of [@friedman2001greedy]: 
  - Fit a _learner_ to $-\mathbf{g}_m$ $\rightarrow \hat{b}_m(x; \gamma_m)$ 
  - Forward stagewise step: $$f_m(x) = f_{m-1}(x) + \rho_m \hat{b}_m(x; \gamma_m)$$

## XGBoost

- Explicit regularization in loss function
- Newton boosting:
  - $\mathbf{h}_m$ is $$-\rho_m \mathbf{H}_m^{-1}\mathbf{g}_m$$
- Custom split point identification: "Weighted Quantile Sketch"
- Computationally optimized

## Learn more
:::::::::::::: {.columns}
::: {.column width="50%"}
[Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) 

![](./../static/esl-cover.jpg){height=400px}
:::
::: {.column width="50%"}
[Trevor Hastie - Gradient Boosting Talk (2014)](https://youtu.be/wPqtzj5VZus)

![](./../static/trevor-hastie-gradient-boosting-thumbnail.jpg)
:::
::::::::::::::



## References {.allowframebreaks}
::: {#refs}
:::