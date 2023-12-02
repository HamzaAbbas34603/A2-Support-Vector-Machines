from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import timeit
import math
import random
import matplotlib.pyplot as plt
from params_generator import BP_params
from sklearn.neural_network import MLPClassifier
from utils import test_algorithm, plot_results
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score

# Excecute the backpropagation using MLPClassifier which is a classification with BP, passing all params printing percentage of error and mean squared error
def execute_bp(dataset, testset, best_bp_params, y_test_scaled, s_min,s_max,bp_test, setname):
    start = timeit.default_timer()
    regr = MLPClassifier(hidden_layer_sizes=list(map(int,best_bp_params[1].split(','))), random_state=1, max_iter=int(best_bp_params[2]), momentum=float(best_bp_params[3]), learning_rate_init=float(best_bp_params[4]),activation=best_bp_params[0])
    regr.fit(dataset.xtable, dataset.ytable)
    stop = timeit.default_timer()
    print('#####  BP  #####\tTime (s): ', round(stop - start,2))
    y_pred_scaled = regr.predict(testset.xtable)
    y_pred_scaled = list(y_pred_scaled)
    y_pred,result = test_algorithm(testset,y_pred_scaled, y_test_scaled, s_min, s_max, bp_test, setname)
    print("Percentage of error over the BP TestSet: {}".format(result))
    print("Mean squared error: {}".format(mean_squared_error(y_test_scaled,y_pred_scaled)))
    plot_results(testset, y_pred)

# Plot ROC for the backpropagation using tensorflow (not used in the main algorithm used once to make the report document)
def plot_rocs_bp(dataset,testset,y_test_scaled):
    model = Sequential()
    model.add(Dense(12, input_shape=(dataset.nf,), activation='sigmoid'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(dataset.xtable, dataset.ytable, epochs=100, verbose=0)
    y_pred, accuracy = model.evaluate(testset.xtable, testset.ytable)
    y_pred = model.predict(testset.xtable)
    print('Accuracy: %.2f' % (accuracy*100))
    y_pred = [i[0] for i in y_pred]
    fpr, tpr, _ = roc_curve(y_test_scaled, y_pred)
    auc = roc_auc_score(y_test_scaled, y_pred)
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
