---
title: A _Blast_ from the _Past_
subtitle: _Fast_ Text Tagging with Conditional Random Fields
title-slide-attributes:
    data-background-image: ./static/crf.svg
    data-background-size: 80%
    data-background-opacity: 0.2
date: TBD
description: "Conditional Random Field models, which where big in the early 2000's, are light weight and fast when it comes to sequence tagging. In this talk we investigate how they stack up against classical Transformers and LLMs, both in terms of accuracy and speed."
header-includes: |
  <meta property="og:url" content="https://duffau.github.io/talks/sequence-tagging/">
  <meta property="og:type" content="website">
  <meta property="og:image" content="https://duffau.github.io/talks/sequence-tagging/static/crf.svg" />
  <meta property="og:title" content="CRF's a Blast from the Past" />
  <meta property="og:description" content=""Conditional Random Field models, which where big in the early 2000's, are light weight fast when it comes to sequence tagging. In this talk we investigate how they stack up against classical Transformers and LLMs, both in terms of accuracy and speed." />
  
  <meta name="twitter:card" content="summary_large_image">
  <meta property="twitter:domain" content="duffau.github.io">
  <meta property="twitter:url" content="https://duffau.github.io/talks/sequence-tagging/">
  <meta name="twitter:title" content="CRF's a Blast from the Past">
  <meta name="twitter:description" content=""Conditional Random Field models, which where big in the early 2000's, are light weight fast when it comes to sequence tagging. In this talk we investigate how they stack up against classical Transformers and LLMs, both in terms of accuracy and speed.">
  <meta name="twitter:image" content="https://duffau.github.io/talks/sequence-tagging/static/crf.svg">
---

###

:::::::::::::: {.columns}
::: {.column width="50%"}
[![](./static/alipes-logo.svg){height=20%}](https://www.alipes.dk)
:::
::: {.column width="50%"}
- Lorem
- Ipsum
- Dolor
- Sit
- [careers.alipes.dk](https://careers.alipes.dk/) 
:::
::::::::::::::


### Formalities

- Slides: [duffau.github.io/talks/sequence-tagging](https://duffau.github.io/talks/sequence-tagging)
- Pizza: 
- Beverages:
- Cocktails: 
- Questions:


---

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


### Big-O reminder

<div class="callout callout-blue">
  <h4 >Definition </h4>
  $f(n) = O(n)$ if
  $f(n) \leq C\cdot n \qquad \text{for all} \quad n>n_0$
</div>


## References {.allowframebreaks}
::: {#refs}
:::