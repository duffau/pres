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
  <meta property="og:description" content="Conditional Random Field models, which where big in the early 2000's, are light weight fast when it comes to sequence tagging. In this talk we investigate how they stack up against classical Transformers and LLMs, both in terms of accuracy and speed.">
  
  <meta name="twitter:card" content="summary_large_image">
  <meta property="twitter:domain" content="duffau.github.io">
  <meta property="twitter:url" content="https://duffau.github.io/talks/sequence-tagging/">
  <meta name="twitter:title" content="CRF's a Blast from the Past">
  <meta name="twitter:description" content="Conditional Random Field models, which where big in the early 2000's, are light weight fast when it comes to sequence tagging. In this talk we investigate how they stack up against classical Transformers and LLMs, both in terms of accuracy and speed.">
  <meta name="twitter:image" content="https://duffau.github.io/talks/sequence-tagging/static/crf.svg">
include-before: |
  <div class="watermark">
    <a href="https://alipes.dk/" rel="noopener noreferrer">
      <img
        src="static/alipes-logo.svg"
      />
    </a>
  </div>
---

### About Me

- Senior Machine Learning Scientist at Alipes
  - News Trading Algorithms
  - Applied NLP and ML
- Co-founded a NLP-powered Legal Tech start-up
- Msc. in Economics from University of Copenhagen
  - Fell into the "$Math$ and `Code`" pot at University

### Slides

![](./static/talk-url-qr-code.svg){ width=40% }
[duffau.github.io/talks/sequence-tagging][3]

---

### Sequence Tagging

```txt
The price of the [Pizza Margherita] is [10 dollars]. 
                  FOOD                  AMOUNT
```

###  
#### Named Entity Recognition (NER) 
```txt
Jim   worked at    Acme Corp. near the beautiful London Bridge.
PER   O      O     ORG  ORG   O    O   O         LOC    LOC   EOS
```

#### Part-of-Speech (POS)
```txt
Jim   worked at    Acme Corp. near the beautiful London Bridge.
NOUN  VERB   PREP  NOUN NOUN  PREP DET ADJ       NOUN   NOUN  EOS
```

$$\begin{aligned}
\text{Labels}:\quad \mathbf{y} &= \{y_1, y_2, \ldots, y_T\}\\
\text{Features}:\quad \mathbf{x} &= \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T\}
\end{aligned}$$

### History

### Recent benchmarks

<div class="mermaid">
  <pre>
    %%{init: {'theme': 'dark', 'themeVariables': { 'darkMode': true }}}%%
    xychart-beta
        title "CoNLL 2003 Usage in Published Papers (paperswithcode.com)"
        x-axis [2020, 2021, 2022, 2023, 2024]
        y-axis "F1-score" 0 --> 100
        bar [30, 35, 30, 40, 45, 50, 51, 51, 50, 54, 55, 56]
  </pre>
</div>

### Papers Using Datasets - NER + POS

<div style="height:400px">
<canvas data-chart="line" >
<!--
{
 "data": {
  "labels": [2020,2021,2022,2023,2024],
  "datasets":[
   {
    "data":[70,92,95,107,50],
    "label":"NER - CoNLL 2003","backgroundColor":"rgba(20,220,220,.8)"
   },
   {
    "data":[86,88,43,41,18],
    "label":"POS - Penn Treebank","backgroundColor":"rgba(220,120,120,.8)"
   }
  ]
 },
 "options": {
        "scales": {
            "y": {
                "display": true,
                "title": {
                    "display": true,
                    "text": "Published Papers" 
                }
            }
        }
    }
}
-->
</canvas>
</div>


### Papers Using Datasets - Abstract tasks

<div style="height:400px">
<canvas data-chart="line" >
<!--
{
 "data": {
  "labels": [2020,2021,2022,2023,2024],
  "datasets":[
   {
    "data":[114,204,216,263, 264],
    "label":"QA - Natural Questions (google + Wiki)","backgroundColor":"rgba(20,220,220,.8)"
   },
   {
    "data":[22,83,122,258,243],
    "label":"Language Modelling - Colossal Clean Crawled Corpus","backgroundColor":"rgba(220,120,120,.8)"
   }
  ]
 },
 "options": {
        "scales": {
            "y": {
                "display": true,
                "title": {
                    "display": true,
                    "text": "Published Papers" 
                }
            }
        }
    }
}
-->
</canvas>
</div>


### Theory



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