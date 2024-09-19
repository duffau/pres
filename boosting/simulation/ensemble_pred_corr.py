import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from adaboost import AdaBoost as AdaBoostM1
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

SEED = 123
N_TREES = 400

def error_rate(y, y_pred):
    return (y != y_pred).sum()/len(y)

def consecutive_prediction_corr(ensemble, X_train):
    y_pred_prev = np.zeros(y_train.shape)
    for estimator in ensemble.estimators_:
        y_pred = estimator.predict(X_train)
        corr = np.corrcoef(y_pred_prev, y_pred)[0,1]
        y_pred_prev = y_pred
        yield corr


X = np.load("data/X.npy")
y = np.load("data/y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = SEED) 

clf = AdaBoostClassifier(n_estimators=N_TREES)
clf.fit(X_train, y_train)
adaboost_pred_corr = [_corr for _corr in consecutive_prediction_corr(clf, X_train)]
avg_adaboost_pred_corr = np.nanmean(adaboost_pred_corr)
print(f"AdaBoost avg. prediction correlation: {avg_adaboost_pred_corr:.3g}")

clf = RandomForestClassifier(n_estimators=N_TREES, max_depth=1, bootstrap=False)
clf.fit(X_train, y_train)
randomforest_pred_corr = [_corr for _corr in consecutive_prediction_corr(clf, X_train)]
avg_randomforest_pred_corr = np.nanmean(randomforest_pred_corr)
print(f"Random forest avg. prediction correlation: {avg_randomforest_pred_corr:.3g}")

clf = BaggingClassifier(n_estimators=N_TREES, bootstrap=True, max_samples=0.25)
clf.fit(X_train, y_train)
bagging_pred_corr = [_corr for _corr in consecutive_prediction_corr(clf, X_train)]
avg_bagging_pred_corr = np.nanmean(bagging_pred_corr)
print(f"Bagging avg. prediction correlation: {avg_bagging_pred_corr:.3g}")

fig1, ax = plt.subplots()
ax.scatter(range(N_TREES), adaboost_pred_corr, c="tab:blue", label=f"Adaboost, avg = {avg_adaboost_pred_corr:.2g}", s=5)
ax.hlines(y=avg_adaboost_pred_corr, xmin=0, xmax=N_TREES, color="tab:blue")
ax.scatter(range(N_TREES), randomforest_pred_corr, c="tab:red", label=f"Random forest, avg = {avg_randomforest_pred_corr:.2g}", s=5)
ax.hlines(y=avg_randomforest_pred_corr, xmin=0, xmax=N_TREES, color="tab:red")
ax.scatter(range(N_TREES), bagging_pred_corr, c="tab:green", label=f"Bagging, avg = {avg_bagging_pred_corr:.2g}", s=5)
ax.hlines(y=avg_bagging_pred_corr, xmin=0, xmax=N_TREES, color="tab:green")
plt.title("Correlation of predictions from consecutive iterations")
plt.ylabel("Correlation")
plt.xlabel("Training iteration")
plt.legend()
plt.savefig("plots/consecutive_predictions_corr.svg")
plt.close()
