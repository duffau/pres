---
title: A _Blast_ from the _Past_
subtitle: _Fast_ Text Tagging with Conditional Random Fields
title-slide-attributes:
	data-background-image: ./static/crf.svg
	data-background-size: 80%
	data-background-opacity: 0.2
date: 2024-09-05
gitdescription: "Conditional Random Field models, which where big in the early 2000's, are light weight and fast when it comes to sequence tagging. In this talk we investigate how they stack up against classical Transformers and LLMs, both in terms of accuracy and speed."
header-includes: |
  <meta property="og:url" content="https://duffau.github.io/talks/sequence-tagging/">
  <meta property="og:type" content="website">
  <meta property="og:image" content="https://duffau.github.io/talks/sequence-tagging/static/crf.svg" />
  <meta property="og:title" content="CRF's a Blast from the Past" />
  \<meta property="og:description" content=""Conditional Random Field models, which where big in the early 2000's, are light weight fast when it comes to sequence tagging. In this talk we investigate how they stack up against classical Transformers and LLMs, both in terms of accuracy and speed." /\>
  
  <meta name="twitter:card" content="summary_large_image">
  <meta property="twitter:domain" content="duffau.github.io">
  <meta property="twitter:url" content="https://duffau.github.io/talks/sequence-tagging/">
  <meta name="twitter:title" content="CRF's a Blast from the Past">
  \<meta name="twitter:description" content=""Conditional Random Field models, which where big in the early 2000's, are light weight fast when it comes to sequence tagging. In this talk we investigate how they stack up against classical Transformers and LLMs, both in terms of accuracy and speed."\>
  <meta name="twitter:image" content="https://duffau.github.io/talks/sequence-tagging/static/crf.svg">
---

\#\#\#

:::::::::::::: {.columns}
::: {.column width="50%"}
[![][image-1]{height=20%}][1]
:::
::: {.column width="50%"}
- Lorem
- Ipsum
- Dolor
- Sit
- [careers.alipes.dk][2] 
:::
::::::::::::::


### Formalities

- Slides: [duffau.github.io/talks/sequence-tagging][3]
- Pizza: 
- Beverages:
- Cocktails: 
- Questions:


---
## Outline (Not presented)

### Introduction and CRF theory

- Sequence tagging task and evolution
	- What is sequence tagging (NER example)
	- Origins in NER and POS using rule-based systems (find examples)
	- Introduction of statistical sequence tagging with HMM, MEMM and CRF
	- Neural net and LSTM revolution
	- Transformers
- Conditional Random Field model - Quick overview
  - Discriminative vs Generative models
	- Examples:
	- Simple discriminative (logistic regression)
	- Simple Generative (AR(1) model)
- CRF theory
	- p(x1, x2, x3) as a directed graph
	- Markov Random Fields (MRF)
	- CRF as extension to MRF
	- Maybe HMM as special case of CRF
- Practical use of Python CRF suite with regex indicator variables in code
- **LLM Accuracy comparison**
  - Intro to sequence performance metrics
- Show performance on
	- Syntatic task (CoNLL-2003 NER task)
	- Semantic task
	- Pragmatic task
	- (Hopefully): CRF and LLM are comparable on syntax heavy tasks and have a big difference in semantic/long dependency tasks
- **LLM Speed Comparison**
- Recap of Big-O notation
- Inference in CRF
	- Naive implementation leads to exponential running time
	- Formulate as Dynamic Programming problem
	- Use memoization
	- Voila: Viterbi algorithm runs in linear time
- Transformers
	  - Quadratic
		- Walkthrough of “canonical” transformer from Stanford lecture - Self-attention leads to quadratic inference
	  - Sub-quadratic - Mention a couple of approaches
- T(n) i.e. actual running time 
- Speed comparison on comparable implementations
	  - Maybe C++ vs Python comparison - Maybe Python only comparison
- **Conclusion**: Quadratic transformers are MUCH slower, Sub-quadratic transformers are also MUCH slower. The constant in front of n actually matters!



### History

<div class="mermaid">
  <pre>
    %%{init: {'theme': 'dark', 'themeVariables': { 'darkMode': true }}}%%
    xychart-beta
        title "Sequence tagging Performance"
        x-axis [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011]
        y-axis "F1-score" 0 --> 100
        line [30, 35, 30, 40, 45, 50, 51, 51, 50, 54, 55, 56]
  </pre>
</div>


### Theory
<div class="mermaid">
  <pre>
    %%{init: {'theme': 'dark', 'themeVariables': { 'darkMode': true }}}%%
    flowchart TD
        x1((Word 1))---y1[ ]-->x2((Label))
        x3((Word 2))---y2[ ]-->x4((Label))
        x5((Word 3))---y3[ ]-->x6((Label))
  </pre>
</div>


### Big-O reminder

<div class="callout callout-blue">
  <h4 >Definition </h4>
  $f(n) = O(n)$
  $\\[10pt]$
  $\text{if} \quad f(n) \leq C\cdot n \qquad \text{for all} \quad n>n_0.$
</div>


## References {.allowframebreaks}
::: {#refs}
:::

[1]:	https://www.alipes.dk
[2]:	https://careers.alipes.dk/
[3]:	https://duffau.github.io/talks/sequence-tagging

[image-1]:	./static/alipes-logo.svg