---
title: "Boosting"
author: Christian Duffau-Rasmussen
date: 03-10-2021
---

## Boosting - What is it ?


> A procedure to combine many __weak__ learners to produce a powerful __committee__. 
>
> [@friedman2009elements, sec. 10.1]

::: notes

- An ensemble method in Machine Learning
- A way of combining so-called weak learner
- into a powerful comittee i.e. by some means og aggregation or averaging

:::

## A brief history


:::::::::::::: {.columns}
::: {.column width="66%"}
![](timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](static/leslie-valiant.jpg){height=130px}

[@valiant1984theory]

- Probably Approximately Correct Learner (PAC)  
- A formal theory of learnability 
- Proof of some broad classes of non-trivial boolean functions are learnable

:::
::::::::::::::

::: notes

- Valiant (1984) - A theory of the learnable
- Defines a mathematical framework for analyzing what classes of problems are learnable in polynomial time. 
- Introduces the Probably Approximately Correct Learner (PAC-learner). 
- Foundation of the field of computational learning theory.

:::

## Side note: PAC learning

- A _concept_ is Probably Approximately Learnable
  -  _if_ in polynomial time an _algorithm_ can deduce a _hypothesis_
  - with an error-rate $<\varepsilon$ with probability at least $1-\delta$
  - $s$ is the size of the _concept_ in bits, encoded appropriately
  - _for all_ $\varepsilon>0$ and $\delta \leq 1$

- The _learning protocol_ is:
  -  learn from examples asking an ORACLE
  -  the ORACLE returns a random example in _unit_ time (O(1))

## A brief history


:::::::::::::: {.columns}
::: {.column width="66%"}
![](timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![Leslie Valiant](static/leslie-valiant.jpg){height=130px}
![Michael Kearns](static/michael-kearns.jpg){height=130px}

[@kearns1989crytographic]

- Introduce _weak learner_  
- Performs only slightly better than chance
- _Hypothesis boosting problem_: weak <=> strong ?

:::
::::::::::::::

::: notes
- The 1989 Crypto paper show that
- If either: 
  - Boolean formulae
  - Deterministic finite automata 
  - Constant-depth threshold circuits 
- are learnable cryptography is toast.

- Kearns and Valiant state as an open problem:
  - Can weak learners be "boosted" into strong learners?
  - I.e. can an algorithm transform weak leaners in to strong ones
  - The notion at the the time was "probably not" 

:::

## Side note: _weak_ and _strong_ learners

- _strongly learnable_ == PAC learnable

- A _concept_ is _weakly_ Learnable
  -  _if_ in polynomial time an _algorithm_ can deduce a _hypothesis_
  - with an error-rate $<\frac{1}{2} - 1/p(s)$ with probability at least $1-\delta$
  - _for all_ $0 < \delta \leq 1$
  - where $s$ is the size of the _concept_ in bits, encoded appropriately

## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](static/robert-schapire.jpg){height=130px}

[@schapire1990strength]

- Cracks the _Hypothesis boosting problem_
- Shows _weak learner_ <=> _strong learner_ 
- An algorithm constructing a strong learner from weak ones 🤯 

:::
::::::::::::::

## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](static/yoav-freund.png){height=130px}

[@freund1990majority]

- Implements a much more efficient _boosting_ algorithm
- Trains learners on weighted subsets of the data 
- Uses majority voting to predict

:::
::::::::::::::

## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](static/yoav-freund.png){height=130px}
![](static/robert-schapire.jpg){height=130px}

[@schapire1995decision]

- Introduces AdaBoost
- First practical boosting algorithm  
- Has been very effective

:::
::::::::::::::


## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](static/jerome-h-friedman.jpeg){height=130px}

[@friedman2001greedy]
[@mason1999boosting]

- Generalizes the _boosting_ concept 
- Describes _boosting_ as gradient descent in function space  

:::
::::::::::::::

## A brief history

:::::::::::::: {.columns}
::: {.column width="66%"}
![](timeline/boosting_timeline.svg){width=1200px}
:::

::: {.column width="33%"}
![](static/tianqi-chen.jpg){height=130px}

[@chen2015higgs]

- Wins Kaggle contest on Higgs Boson using XGBoost 
- XGBoost quickly becomes the most winning algorithm  

:::
::::::::::::::

## Ensemble methods

- Bagging: Grow trees using _random subsets of data (with replacement)_
- Random forest: Grow trees using _random subset of features_
- Boosting: Grow trees on _re-weighted dataset_

. . . 

$$\text{Boosting} \succ \text{Random forest} \succ \text{Bagging} \succ \text{Tree}$$

## Simulation example

$$Y = \begin{cases}
1 & \text{if}\quad X_1^2 + \ldots + X_{10}^2 > 9.34 \\
-1 & \text{else}
\end{cases}\quad X_i\sim N(0,1)$$

![Simulated 10-D nested spheres](experiment/gen_data.svg){height=400px}

## Simulation example

![](experiment/ensemble_test_errors.svg)


## Bias/variance trade-off

$$\text{MSE} = \text{Bias}(\hat{f})^2 + \text{Var}(\hat{f}) + \text{Irreducible noise}$$

## Variance of averages

$$\text{Var}\left(\frac{1}{n}\sum_i^n X_i\right) = \frac{1}{n}\sum_i^n \text{Var}\left(X_i\right) + \frac{1}{n}\sum_{i\neq j} \text{Cov}(X_i, X_j)$$

. . .

$$\text{Variance of ensemble} = \frac{\text{Var(Trees)}}{n} + \frac{\text{Cov( Trees)}}{n}$$

## Variance and bias reduction

![Classifying 10-D nested spheres](experiment/consecutive_predictions_corr.svg)


## Boosting

![](static/Ensemble_Boosting.svg)

::: notes

- A collection of _weak learners_ (e.g. classifier) are trained sequentially.
- Each _learner_ is trained on the same dataset.
- Each example is re-weighted in each iteration
- Poorly predicted examples get higher weight 
- Well predicted examples get lower weight
:::


## Forward Stagewise Additive Learning

1. Initialize $f_0(x) = 0$
2. For $m=1$ to $M$:
   a. Compute $$(\beta_m, \gamma_m) = \underset{\beta, \gamma}{\text{argmin}} \sum_{i=1}^N L(y_i, f_{m-1}(x_i)+ \beta b(x_i;\gamma))$$
   b. Set $f_m(x) = f_{m-1}(x) + \beta_m b(x; \gamma_m)$

::: notes

- Boosting is a special case of _Forward Stagewise Additive Learning_
- It's an approximation technique for general additive models

:::
## Adaboost

1. Initialize weights $w_i=1/N$
2. For $m=1$ to $M$:
   1. Fit classifier $G_m(x)$ to training data using $w_i$'s
   2. Compute $$\text{err} = \frac{\sum_{i=1}^N w_i I(y_i \neq G_m(x_i))}{\sum_{i=1}^N w_i}$$
   3. Compute $\alpha_m = \log((1-\text{err})/\text{err})$ 
   4. Set $w_i \leftarrow \exp(\alpha_m I(y_i \neq G_m(x_i)))$
3. Output $$G(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m G_m(x)\right)$$


## Adaboost

- Not originally motivated in _Forward Stagewise Additive Learning_
- One can show that Adaboost is a stagewise additive model with loss $$L(y, f (x)) = \exp(−y f(x))$$ where $y\in \{-1, 1\}$ and $f(x) \in \mathbb{R}$.
- Usually for classification we use cross-entropy $$L(y, f(x)) = y^\prime \log p(x) + (1-y^\prime) \log(1- p(x)) $$ where $y^\prime = (y+1)/2\in \{0,1\}$ and $p(x)$ is the softmax function.
- In theory the two loss functions are equivalent
- On average they produce the same _model_ $f$

## Adaboost

- For finite samples the exponential loss has drawbacks
- To much weight is given to errors

. . .

![Exponential loss and cross-entropy](plots/loss_functions.svg){height=300px}

## Adaboost

- Adaboost has trouble on noisy data

. . .

:::::::::::::: {.columns}
::: {.column width="50%"}
![Test error clean data](experiment/ensemble_test_errors.svg)
:::
::: {.column width="50%"}
![Test error noisy data](experiment/ensemble_test_errors_noisy.svg)
:::
::::::::::::::


:::notes
- The deviance yf is only negative if they have different signs
- So all negative values are errors
- The exponential loss eight these error much heavier
:::

## Adaboost

- How to fix Adaboost?
- Can we easily plug in a more robust loss function? 

. . .

::: nonincremental
1. Initialize weights $w_i=1/N$
2. For $m=1$ to $M$:
  1. Fit classifier $G_m(x)$ to training data using $w_i$'s
  2. Compute $\text{err} = \frac{\sum_{i=1}^N w_i I(y_i \neq G_m(x_i))}{\sum_{i=1}^N w_i}$
  3. Compute $\alpha_m = \log((1-\text{err})/\text{err})$ 
  4. Set $w_i \leftarrow \exp(\alpha_m I(y_i \neq G_m(x_i)))$
3. Output $$G(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m G_m(x)\right)$$
:::

🤷‍♂️

## Gradient boosting 

- Introduced by [@friedman2001greedy] as a new view of boosting
- Makes it possible to _derive_ a boosting procedure
- Solves each forward stagewise step as _gradient descent_ in a function space
- Applies to any differentiable loss function


## Side note: Gradient boosting in SciKit Learn

_Scikit-learn documentation_:

> GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.

. . .

Sound promising...

. . .

> `loss: {‘deviance’, ‘exponential’}`
>
> For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm.

. . .

🤦‍♂️

## Gradient boosting




## XGBoost

## References {.allowframebreaks}
::: {#refs}
:::