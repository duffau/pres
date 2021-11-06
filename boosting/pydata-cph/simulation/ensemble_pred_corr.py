import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from adaboost import AdaBoost as AdaBoostM1
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
SEED = 123
N_TREES = 400
def error_rate(y, y_pred):
    return (y != y_pred).sum()/len(y)

X = np.load("X.npy")
y = np.load("y.npy")

# Train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = SEED) 

clf = AdaBoostClassifier(n_estimators=N_TREES)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print(f"Final training error rate:{ error_rate(y_train, y_pred)}")
y_pred = clf.predict(X_test)
print(f"Final test error rate:{ error_rate(y_test, y_pred)}")

def consecutive_prediction_corr(ensemble, X_train):
    y_pred_prev = np.zeros(y_train.shape)
    for estimator in ensemble.estimators_:
        y_pred = estimator.predict(X_train)
        # print(y_pred)
        # print(y_pred_prev)
        corr = np.corrcoef(y_pred_prev, y_pred)[0,1]
        y_pred_prev = y_pred
        yield corr
 
adaboost_pred_corr = [_corr for _corr in consecutive_prediction_corr(clf, X_train)]

clf = RandomForestClassifier(n_estimators=N_TREES, max_depth=1, bootstrap=False)
clf.fit(X_train, y_train)
randomforest_pred_corr = [_corr for _corr in consecutive_prediction_corr(clf, X_train)]

clf = BaggingClassifier(n_estimators=N_TREES, bootstrap=True, max_samples=0.25)
clf.fit(X_train, y_train)
bagging_pred_corr = [_corr for _corr in consecutive_prediction_corr(clf, X_train)]

fig1, ax = plt.subplots()
ax.scatter(range(N_TREES), adaboost_pred_corr, c="tab:blue", label="Adaboost", s=5)
ax.scatter(range(N_TREES),randomforest_pred_corr, c="tab:red", label="Random forest", s=5)
ax.scatter(range(N_TREES),bagging_pred_corr, c="tab:green", label="Bagging", s=5)
plt.title("Correlation of predictions from consecutive iterations")
plt.ylabel("Correlation")
plt.xlabel("Training iteration")
plt.legend()
plt.savefig("plots/consecutive_predictions_corr.svg")
plt.close()
