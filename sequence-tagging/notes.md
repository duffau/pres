## Outline

### Introduction and CRF theory

- Sequence tagging task and evolution
	- What is sequence tagging (NER example)
	- Origins in NER and POS using rule-based systems (find examples)
	    - NER History: https://wandb.ai/madhana/Named_Entity_Recognition/reports/A-Beginner-s-Guide-to-Named-Entity-Recognition-NER---VmlldzozNjE2MzI1
    	- B. Aldelberg. Nodose: A tool for semi-automaticallyextracting structured and semistructured data fromtext documents. In SIGMOD, 1998
    	- Ralph Grishman and Beth Sundheim. 1996. Message Understanding Conference- 6: A Brief History.: https://aclanthology.org/C96-1079/ 
	- Introduction of statistical sequence tagging with HMM, MEMM and CRF
	- Neural net and LSTM revolution
	- First gen Transformers (BERT, RoBERTa, Electra)
	- LLM's. (GPT-2, GPT-3, Claude, LLaMA) [Stanford 2024 - Transformers Slide 9][14]
- Conditional Random Field model - Quick overview

### Historical perormance for NER and POS

- https://nlpprogress.com/english/named_entity_recognition.html
- https://nlpprogress.com/english/part-of-speech_tagging.html
- https://rajpurkar.github.io/SQuAD-explorer/
- https://aclweb.org/aclwiki/POS_Tagging_(State_of_the_art)

### Introduction and CRF theory

- Discriminative vs Generative models [Bishop chap. 4.3 + 4.4][1]
	- Examples:
    	- Simple discriminative (logistic regression)
    	- Simple Generative (AR(1) model)
  	- On Discriminative vs. Generative Classifier: A comparison logistisc regression and naive bayes
- CRF theory
	- p(x1, x2, x3) as a directed graph [Bishop chap. 8.1][2]
	- Markov Random Fields (MRF) [Bishop chap. 8.3][3]
	- CRF as extension to MRF
	- Maybe HMM as special case of CRF [Sutton - Intro to CRF][4]
- Practical use of Python CRF suite with regex indicator variables in code

### LLM Accuracy comparison

- Intro to sequence performance metrics
- Show performance on
	- Syntatic driven task (CoNLL-2003 NER task)
	- Semantic driven task
	- [Pragmatics][5] driven task
	- (Hopefully): CRF and LLM are comparable on syntax heavy tasks and have a big difference in performance on semantic and pragmatic information driven tasks

### LLM Speed Comparison

- Recap of Big-O notation
- Inference in CRF
	- Naive implementation leads to exponential running time
	- Formulate as Dynamic Programming problem
	- Use memoization
	- Voila: Viterbi algorithm runs in linear time ([Hugo Larochelle Slides][6])

### LLM Speed Comparison

- Transformers
	- Quadratic
	- Walkthrough of “canonical” transformer from [Stanford lecture notes][12] 
        - Self-attention leads to quadratic inference [Stanford CS224n 2024 - lecture  slides 18, 58][14]
	- Sub-quadratic - Mention a couple of approaches (Local windows, global windows, random interactions [Stanford CS224n 2024 - lecture  slide 59,60][14])
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
- CoNLL 2003 dataset: [https://huggingface.co/datasets/eriktks/conll2003][7]

- [A survey on recent advances in Named Entity Recognition - Keraghel (2024)][8]
- [Pattern recognition and machine learning - Bishop & Nasrabadi (2006)][9]
- [Neural Networks Course - Larochelle][10]
- [An introduction to conditional random fields - Sutton & McCallum (2012)][11]
- [Attention is All You Need - Vaswani et. al. (2017)][15]
- [Stanford CS224n (2023): Natural Language Processing with Deep Learning - Note 10: Self-Attention & Transformers][12]
- [Stanford CS224n (2023): Natural Language Processing with Deep Learning - Slides: Self-Attention & Transformers][13]
- [Stanford CS224n (2024): Natural Language Processing with Deep Learning - Slides: Self-Attention & Transformers][14]

[1]:	http://crowley-coutaz.fr/jlc/Courses/2020/PRML/ENSI3.PRML.S6.Encoders.pdf
[2]:	http://crowley-coutaz.fr/jlc/Courses/2020/PRML/ENSI3.PRML.S6.Encoders.pdf
[3]:	http://crowley-coutaz.fr/jlc/Courses/2020/PRML/ENSI3.PRML.S6.Encoders.pdf
[4]:	https://www.nowpublishers.com/article/DownloadSummary/MAL-013
[5]:	https://en.wikipedia.org/wiki/Pragmatics
[6]:	https://larocheh.github.io/ift725/3_04_computing_partition_function.pdf
[7]:	https://huggingface.co/datasets/eriktks/conll2003
[8]:	https://arxiv.org/html/2401.10825v1
[9]:	http://crowley-coutaz.fr/jlc/Courses/2020/PRML/ENSI3.PRML.S6.Encoders.pdf
[10]:	https://larocheh.github.io/neural_networks/content.html
[11]:	https://www.nowpublishers.com/article/DownloadSummary/MAL-013
[12]:   https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf
[13]:   https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture08-transformers.pdf
[14]:   https://web.stanford.edu/class/cs224n/slides/cs224n-spr2024-lecture08-transformers.pdf
[15]:   https://arxiv.org/pdf/1706.03762

