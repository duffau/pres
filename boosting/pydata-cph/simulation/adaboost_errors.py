import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

SEED = 12345
N_TREES = 400
REPS = 1

def error_rate(y, y_pred):
    return (y != y_pred).sum()/len(y)

training_errors = []
test_errors = []

x_path = "data/X.npy"
y_path = "data/y.npy"

X = np.load(x_path)
y = np.load(y_path)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size = 0.8, 
    random_state = SEED
) 

clf = AdaBoostClassifier(n_estimators=N_TREES, learning_rate=1.0, algorithm="SAMME.R")

clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print(f"Final training error rate:{ error_rate(y_train, y_pred)}")
y_pred = clf.predict(X_test)
print(f"Final test error rate:{ error_rate(y_test, y_pred)}")

training_errors = [error_rate(y_train, y_pred) for y_pred in clf.staged_predict(X_train)]
test_errors = [error_rate(y_test, y_pred) for y_pred in clf.staged_predict(X_test)]

fig1, ax = plt.subplots()
ax.plot(training_errors, c="tab:blue", label="Training error")
ax.plot(test_errors, c="tab:red", label="Test error")
plt.legend()
plt.title("AdaBoost")
plt.ylabel("Misclassification error rate/Loss")
plt.xlabel("Boosting iteration")
plt.tight_layout()
plt.savefig("plots/adaboost_errors.svg")
plt.close()

