import numpy as np
from numpy.core.fromnumeric import argmax
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

SEED = 123
MAX_ESTIMATORS = 200
SINGLE_TREE = "Single tree"
BAGGING = "Bagging with full trees"
RANDOM_FOREST = "Random forest with full trees"
ADABOOST = "AdaBoost with stumps"


def error_rate(y, y_pred):
    return (y != y_pred).sum()/len(y)

suffixes = ["", "noisy"]

def join_underscore(*args):
    return "_".join(str(arg) for arg in args if arg)

for suffix in suffixes:
    x_path = f"data/{join_underscore('X', suffix) + '.npy'}"
    y_path = f"data/{join_underscore('y', suffix) + '.npy'}"
    print(f"Loading generated data '{x_path}'' and '{y_path}'...")
    X = np.load(x_path)
    y = np.load(y_path)

    # Train - test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = SEED) 


    def incremental_test_error(clf, X_test, y_test):
        test_errors = []
        y_proba = np.zeros(shape=(y_test.shape[0],2))
        for i, estimator in enumerate(clf.estimators_):
            y_proba += (estimator.predict_proba(X_test) - y_proba)/(i+1)
            y_pred = np.argmax(y_proba, axis=1)*2 - 1
            test_errors.append(error_rate(y_test, y_pred))
        return test_errors

    test_errors = {}
    test_errors[SINGLE_TREE] = []
    print("Fitting single tree ...")
    for max_depth in range(1, MAX_ESTIMATORS):
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test_errors[SINGLE_TREE].append(error_rate(y_test, y_pred))


    print("Fitting Bagging classifier ...")
    clf = BaggingClassifier(n_estimators=MAX_ESTIMATORS, bootstrap=True, n_jobs=-1)
    clf.fit(X_train, y_train)
    test_errors[BAGGING] = incremental_test_error(clf, X_test, y_test)

    print("Fitting Random forest ...")
    clf = RandomForestClassifier(n_estimators=MAX_ESTIMATORS, n_jobs=-1)
    clf.fit(X_train, y_train)
    test_errors[RANDOM_FOREST] = incremental_test_error(clf, X_test, y_test)

    print("Fitting AdaBoost ...")
    clf = AdaBoostClassifier(n_estimators=MAX_ESTIMATORS, algorithm="SAMME.R", learning_rate=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_errors[ADABOOST] = incremental_test_error(clf, X_test, y_test)

    plot_filename = f"plots/{join_underscore('ensemble_test_errors', suffix) + '.svg'}"
    print(f"Plotting '{plot_filename}' ...")
    fig1, ax = plt.subplots()
    for algo, _test_errors in test_errors.items():
        ax.plot(_test_errors, label=algo)
    plt.legend()
    plt.title("Test error")
    plt.ylabel("Misclassification error rate")
    plt.xlabel("n trees/n leafs")
    plt.savefig(plot_filename)
    plt.close()
