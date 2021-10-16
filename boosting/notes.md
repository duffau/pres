Tree growing algorithms:
- CART
- ID3
- C4.5
- C5.0

> Trees require pN log N operations for an initial sort for each predictor, and typically another pN log N operations for the split computations. If the splits occurred near the edges of the predictor ranges, this number could increase to N 2 p.

Trevor Hastie and Robert Tibshirani onine lectures on statistical learning:
https://youtu.be/RSWg_islt9c

Trevor Hastie talk on boosting:
https://www.youtube.com/watch?v=wPqtzj5VZus
# Modelling interaction

Boosting generates a collection of trees, yielding an additive model. The depth og the single trees governs the level of interaction between variables.

d = 1: No interaction - Tree stumps
d = 2: 2 variable interaction
d = 3: At most 3 variable interaction

# Scikit Learn

-  AdaBoost implements