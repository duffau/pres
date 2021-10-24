import numpy as np
from numpy.core.fromnumeric import argmax
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

SEED = 123
MAX_ESTIMATORS = 200

def error_rate(y, y_pred):
    return (y != y_pred).sum()/len(y)


X = np.load("X_noisy.npy")
y = np.load("y_noisy.npy")

# Train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = SEED) 

# bag = BaggingClassifier(n_estimators=100)
# random_forest = RandomForestClassifier(n_estimators=400)
# adaboost = AdaBoostClassifier(n_estimators=400, learning_rate=1)

def incremental_test_error(clf, X_test, y_test):
    test_errors = []
    y_proba = np.zeros(shape=(y_test.shape[0],2))
    for i, estimator in enumerate(clf.estimators_):
        y_proba += (estimator.predict_proba(X_test) - y_proba)/(i+1)
        y_pred = np.argmax(y_proba, axis=1)*2 - 1
        test_errors.append(error_rate(y_test, y_pred))
    return test_errors

test_errors = {}
test_errors["single_tree"] = []
print("Fitting single tree ...")
for max_depth in range(1, MAX_ESTIMATORS):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_errors["single_tree"].append(error_rate(y_test, y_pred))


print("Fitting Bagging classifier ...")
clf = BaggingClassifier(n_estimators=MAX_ESTIMATORS, bootstrap=True, n_jobs=-1)
clf.fit(X_train, y_train)
test_errors["bagging"] = incremental_test_error(clf, X_test, y_test)

print("Fitting Random forest ...")
clf = RandomForestClassifier(n_estimators=MAX_ESTIMATORS, n_jobs=-1)
clf.fit(X_train, y_train)
test_errors["random_forest"] = incremental_test_error(clf, X_test, y_test)

print("Fitting AdaBoost ...")
clf = AdaBoostClassifier(n_estimators=MAX_ESTIMATORS)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
test_errors["adaboost"] = incremental_test_error(clf, X_test, y_test)


fig1, ax = plt.subplots()
for algo, _test_errors in test_errors.items():
    ax.plot(_test_errors, label=algo)
plt.legend()
plt.title("Test error")
plt.ylabel("Misclassification error rate")
plt.xlabel("n trees/n leafs")
plt.savefig("ensemble_test_errors_noisy.svg")
plt.close()
