import timeit
import math
import random
import matplotlib.pyplot as plt
from utils import test_algorithm, plot_results
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score
from sklearn import linear_model

# Excecute the backpropagation using LogisticRegression which is a classification for MLR, printing percentage of error and mean squared error
def execute_mlr(dataset, testset, y_test_scaled, s_min,s_max,mlr_test, setname):
    start = timeit.default_timer()
    regr = linear_model.LogisticRegression()
    regr.fit(dataset.xtable, dataset.ytable)
    stop = timeit.default_timer()
    print('#####  MLR  #####\tTime (s): ', round(stop - start,2))
    y_pred_scaled = regr.predict(testset.xtable)
    y_pred_scaled = list(y_pred_scaled)
    y_pred,result = test_algorithm(testset,y_pred_scaled, y_test_scaled, s_min, s_max, mlr_test, setname)
    print("Percentage of error over the MLR TestSet: {}".format(result))
    print("Mean squared error: {}".format(mean_squared_error(y_test_scaled,y_pred_scaled)))
    plot_results(testset, y_pred)

# Plot ROC for the MLR (not used in the main algorithm used once to make the report document)
def plot_roc_mlr(dataset, testset, y_test_scaled):
    regr = linear_model.LinearRegression()
    regr.fit(dataset.xtable, dataset.ytable)
    y_pred_scaled = regr.predict(testset.xtable)
    y_pred = list(y_pred_scaled)
    fpr, tpr, _ = roc_curve(y_test_scaled, y_pred)
    auc = roc_auc_score(y_test_scaled, y_pred)
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()