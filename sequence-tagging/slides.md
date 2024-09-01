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

::: incremental
- Senior Machine Learning Scientist at Alipes
  - News Trading Algorithms
  - Applied NLP and Machine Learning
- Msc. in Economics from University of Copenhagen from 2016
  - Fell into the "$Math$ and `Code`" pot at University
- Before joining Alipes
  - Automated Sports Betting
  - Co-Founded a Machine Learning Consultancy
  - Co-founded a NLP-powered Legal Tech start-up
:::

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

## Evolution of NLP and Sequence tagging 

### 1954-1966 - AI Over-optimism and AI Winter
::: incremental
- 1954: IBM-Georgetown machine translation: Sixty Russian sentences translated into English
- 1957: Noam Chomsky *Syntactic Structures*: 
  - *Generative Grammar*: A system of rules that generate exactly those combinations of words that form grammatical sentences
  - *Anti-probabilistic*: "probabilistic models give no particular insight into some of the basic problems of syntactic structure."
:::

### 1954-1966 - AI Over-optimism and AI Winter
::: incremental
- 1958: H. A. Simon and Allen Newell:"within ten years a digital computer will discover and prove an important new mathematical theorem"
- 1960's: Slow progress in machine translation
- 1966: ALPAC report ledas to defunding of machine translation in US
- 1974–1980: Triggers First AI Winter
:::

### Late 1980's and 1990's - Rise Statistical Models
::: incremental
- Less dominance of Chomskyan theories of linguistics
- More computational power
- Availability of Annotated Datasets
- Give Rise to Statistical NLP
  - 1989: Hidden Markov Models (HMMs) for Speech
  - 1993: Penn Treebank Project
  - 1995: WordNet: A Lexical Database for English
  - 1996: A Maximum Entropy Model for Part-Of-Speech Tagging
  - 2001: Conditional Random Field
:::

### 2000's to 2010 - Neural Models and Word Embeddings

- Neural net and LSTM revolution

### 2010 to Today - Transformers

- First gen Transformers (BERT, RoBERTa, Electra)
- LLM's. (GPT-2, GPT-3, Claude, LLaMA) [Stanford 2024 - Transformers Slide 9][14]



### Papers Using Datasets ^1^

<div style="height:400px">
<canvas data-chart="line">
<!--
{
 "data": {
  "labels": [2020,2021,2022,2023,2024],
  "datasets":[
   {
    "data":[86,88,43,41,18],
    "label":"POS - Penn Treebank",
    "yAxisID": "y",
    "fill": false
   },
   {
    "data":[63,112,105,156, 195],
    "label":"QA - TriviaQA (Wiki + Web)",
    "yAxisID": "y1"
   },
   {
    "data":[16,30,54,160,327],
    "label":"NLI - HellaSwag Sentence Completion",
    "yAxisID": "y1"
   }
  ]
 },
 "options": {
  "scales": {
   "y": {
    "type": "linear",
    "display": true,
    "position": "left",
    "title": {
     "display": true,
     "text": "Published Papers"
    }
   },
   "y1": {
    "type": "linear",
    "display": true,
    "position": "right",
    "title": {
     "display": true,
     "text": "Published Papers"
    },
    "grid": {
     "drawOnChartArea": false
    }
   }
  }
 }
}
-->
</canvas>
</div>

Abstract tasks have taken over lower level tasks

::: footer
^1^ Source: https://paperswithcode.com/datasets
:::

::: notes
- Penn Tree Bank
  - Penn State Tree Bank
  - Initially released in 1992
  - First richly annotated text corpus 
  - 1 mio Annotated tokens (2500 stories) from Wall Street Journal Article from 1989 Wall Street Journal 
  - 2022: Sequence Aligment Ensemble-BART encoder: 98.15 Accuracy
    - Ensemble of BART models 
    - Weighted voting where weights a proportional to avg. alignment score with other predictions in ensemble  
  - 2018: BI-LSTM: 97.96 Accuracy
- TriviaQA: Challenging than QA pairs
  - Long context
  - Answers not optained by span prediction in question or context
  - 2017 University of Washington NLP
  - Claude 5 shots: 87.5 f1 score
  - https://paperswithcode.com/sota/question-answering-on-triviaqa
- HellaSwag: Common sense Natural Language Inference
  - "A woman sits at a piano," -> "She sets her fingers on the keys."
  - Humans have 95% accuracy
  - From Allen Institute for AI a Non-Profit research org.
  - GPT4 10 shots: 95.3 Accuracy
  - https://paperswithcode.com/sota/sentence-completion-on-hellaswag
:::



### Theory

### Conditional Random Field model - Quick overview


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