## Outline

### Introduction and CRF theory

- Sequence tagging task and evolution
	- What is sequence tagging (NER example)
	- Origins in NER and POS using rule-based systems (find examples)
	- Introduction of statistical sequence tagging with HMM, MEMM and CRF
	- Neural net and LSTM revolution
	- Transformers
- Conditional Random Field model - Quick overview

### Introduction and CRF theory

  - Discriminative vs Generative models [Bishop chap. 4.3 + 4.4][2]
	- Examples:
  	- Simple discriminative (logistic regression)
  	- Simple Generative (AR(1) model)
- CRF theory
	- p(x1, x2, x3) as a directed graph [Bishop chap. 8.1][2]
	- Markov Random Fields (MRF) [Bishop chap. 8.3][2]
	- CRF as extension to MRF
	- Maybe HMM as special case of CRF [Sutton - Intro to CRF][5]
- Practical use of Python CRF suite with regex indicator variables in code

### LLM Accuracy comparison

- Intro to sequence performance metrics
- Show performance on
	- Syntatic driven task (CoNLL-2003 NER task)
	- Semantic driven task
	- [Pragmatics](https://en.wikipedia.org/wiki/Pragmatics) driven task
	- (Hopefully): CRF and LLM are comparable on syntax heavy tasks and have a big difference in performance on semantic and pragmatic information driven tasks

### LLM Speed Comparison

- Recap of Big-O notation
- Inference in CRF
	- Naive implementation leads to exponential running time
	- Formulate as Dynamic Programming problem
	- Use memoization
	- Voila: Viterbi algorithm runs in linear time ([Hugo Larochelle Slides][4])

### LLM Speed Comparison

- Transformers
	- Quadratic
	- Walkthrough of “canonical” transformer from Stanford lecture - Self-attention leads to quadratic inference
	- Sub-quadratic - Mention a couple of approaches
- T(n) i.e. actual running time 
- Speed comparison on comparable implementations
	- Maybe C++ vs Python comparison - Maybe Python only comparison

### Conclusion

- CRF's are great at identifying entities which are indetified by syntactic and some extent semantic information
- Quadratic transformers are MUCH MUCH slower, Sub-quadratic transformers are also MUCH slower. The constant in front of n actually matters!


# References

- FiNER: Financial Numeric Entity Recognition for XBRL Tagging
  - Dataset: https://huggingface.co/datasets/nlpaueb/finer-139
  - Paper: https://arxiv.org/abs/2203.06482

- [A survey on recent advances in Named Entity Recognition - Keraghel (2024)][1]
- [Pattern recognition and machine learning - Bishop & Nasrabadi (2006)][2]
- [Neural Networks Course - Larochelle][3]
- [An introduction to conditional random fields - Sutton & McCallum (2012)][5]

[1]: https://arxiv.org/html/2401.10825v1
[2]: http://crowley-coutaz.fr/jlc/Courses/2020/PRML/ENSI3.PRML.S6.Encoders.pdf
[3]: https://larocheh.github.io/neural_networks/content.html
[4]: https://larocheh.github.io/ift725/3_04_computing_partition_function.pdf
[5]: https://www.nowpublishers.com/article/DownloadSummary/MAL-013