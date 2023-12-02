import timeit
from sklearn.svm import SVC
import math
import random
import matplotlib.pyplot as plt
from params_generator import SVM_params
from utils import test_algorithm, plot_results
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score

# Excecute the backpropagation using SVC which is SVM classification, passing all params printing percentage of error and mean squared error
def execute_svm(dataset, testset, best_svm_params, y_test_scaled, s_min,s_max,svm_test, setname):
    start = timeit.default_timer()
    regr = SVC(kernel=best_svm_params[0],C=int(best_svm_params[1]))
    regr.fit(dataset.xtable, dataset.ytable)
    stop = timeit.default_timer()
    print('#####  SVM  #####\tTime (s): ', round(stop - start,2))
    y_pred_scaled = regr.predict(testset.xtable)
    y_pred_scaled = list(y_pred_scaled)
    y_pred,result = test_algorithm(testset,y_pred_scaled, y_test_scaled, s_min, s_max, svm_test, setname)
    print("Percentage of error over the SVM TestSet: {}".format(result))
    print("Mean squared error: {}".format(mean_squared_error(y_test_scaled,y_pred_scaled)))
    plot_results(testset, y_pred)