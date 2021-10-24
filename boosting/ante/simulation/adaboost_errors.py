import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from adaboost import AdaBoost as AdaBoostM1
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

SEED = 123

def error_rate(y, y_pred):
    return (y != y_pred).sum()/len(y)

X = np.load("X.npy")
y = np.load("y.npy")

# Train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = SEED) 

# clf = AdaBoostM1(n_estimators=100)
clf = AdaBoostClassifier(n_estimators=400, learning_rate=1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print(f"Final training error rate:{ error_rate(y_train, y_pred)}")
y_pred = clf.predict(X_test)
print(f"Final test error rate:{ error_rate(y_test, y_pred)}")

training_errors = []
test_errors = []
for y_pred in clf.staged_predict(X_train):
    training_errors.append(error_rate(y_train, y_pred))

for y_pred in clf.staged_predict(X_test):
    test_errors.append(error_rate(y_test, y_pred))

fig1, ax = plt.subplots()
ax.plot(training_errors, c="tab:blue", label="Training error")
ax.plot(test_errors, c="tab:red", label="Test error")
plt.legend()
plt.ylabel("Misclassification error rate")
plt.xlabel("Boosting iteration")
plt.savefig("test_error.svg")
plt.close()
