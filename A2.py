from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score
import argparse
from mlr import *
from bp import *
from svm import *
from utils import *

# Do cross validation all at once
def cross_validation():
    # percetange of the data dedicated to validation
    # in this algorithm we basically take a set from the dataset and train it in order to do validation
    nfold = 4
    correct = 0
    incorrect = 0
    oresult = 0
    for j in range(0,nfold):
        val_set = dataset.ns // nfold
        xtable = [*dataset.xtable[0:j*val_set],*dataset.xtable[j*val_set+val_set:]]
        ytable = [*dataset.ytable[0:j*val_set],*dataset.ytable[j*val_set+val_set:]]
        regr = SVC(kernel=best_svm_params[0],C=int(best_svm_params[1]))
        regr.fit(xtable, ytable)
        y_val_scaled = dataset.ytable[j*val_set:j*val_set+val_set]
        y_pred_scaled = regr.predict(dataset.xtable[j*val_set:j*val_set+val_set])
        y_pred_scaled = list(y_pred_scaled)
        y_pred,result = test_algorithm(dataset,y_pred_scaled, y_val_scaled, s_min, s_max, mlr_test)
        for i in range(0,len(y_pred)):
            if y_pred[i] == y_val_scaled[i]:
                correct += 1
            else:
                incorrect += 1
        oresult += result
    regr = SVC(kernel=best_svm_params[0],C=int(best_svm_params[1]))
    regr.fit(dataset.xtable, dataset.ytable)
    y_test_scaled = testset.ytable
    y_pred_scaled = regr.predict(testset.xtable)
    y_pred_scaled = list(y_pred_scaled)
    y_pred,result = test_algorithm(testset,y_pred_scaled, y_test_scaled, s_min, s_max, mlr_test)
    print("Percentage of error over the SVM Cross-Validation: {} Test: {}. Correct: {} Incorrect: {}".format(oresult/nfold,result,correct,incorrect))

    correct = 0
    incorrect = 0
    oresult = 0
    for j in range(0,nfold):
        val_set = dataset.ns // nfold
        xtable = [*dataset.xtable[0:j*val_set],*dataset.xtable[j*val_set+val_set:]]
        ytable = [*dataset.ytable[0:j*val_set],*dataset.ytable[j*val_set+val_set:]]
        regr = MLPClassifier(hidden_layer_sizes=list(map(int,best_bp_params[1].split(','))), random_state=1, max_iter=int(best_bp_params[2]), momentum=float(best_bp_params[3]), learning_rate_init=float(best_bp_params[4]),activation=best_bp_params[0])
        regr.fit(xtable, ytable)
        y_val_scaled = dataset.ytable[j*val_set:j*val_set+val_set]
        y_pred_scaled = regr.predict(dataset.xtable[j*val_set:j*val_set+val_set])
        y_pred_scaled = list(y_pred_scaled)
        y_pred,result = test_algorithm(dataset,y_pred_scaled, y_val_scaled, s_min, s_max, mlr_test)
        for i in range(0,len(y_pred)):
            if y_pred[i] == y_val_scaled[i]:
                correct += 1
            else:
                incorrect += 1
        oresult += result
    regr = MLPClassifier(hidden_layer_sizes=list(map(int,best_bp_params[1].split(','))), random_state=1, max_iter=int(best_bp_params[2]), momentum=float(best_bp_params[3]), learning_rate_init=float(best_bp_params[4]),activation=best_bp_params[0])
    regr.fit(dataset.xtable, dataset.ytable)
    y_test_scaled = testset.ytable
    y_pred_scaled = regr.predict(testset.xtable)
    y_pred_scaled = list(y_pred_scaled)
    y_pred,result = test_algorithm(testset,y_pred_scaled, y_test_scaled, s_min, s_max, mlr_test)
    print("Percentage of error over the BP Cross-Validation: {} Test: {}. Correct: {} Incorrect: {}".format(oresult/nfold,result,correct,incorrect))

    correct = 0
    incorrect = 0
    oresult = 0
    for j in range(0,nfold):
        val_set = dataset.ns // nfold
        xtable = [*dataset.xtable[0:j*val_set],*dataset.xtable[j*val_set+val_set:]]
        ytable = [*dataset.ytable[0:j*val_set],*dataset.ytable[j*val_set+val_set:]]
        regr = linear_model.LogisticRegression()
        regr.fit(xtable, ytable)
        y_val_scaled = dataset.ytable[j*val_set:j*val_set+val_set]
        y_pred_scaled = regr.predict(dataset.xtable[j*val_set:j*val_set+val_set])
        y_pred_scaled = list(y_pred_scaled)
        y_pred,result = test_algorithm(dataset,y_pred_scaled, y_val_scaled, s_min, s_max, mlr_test)
        for i in range(0,len(y_pred)):
            if y_pred[i] == y_val_scaled[i]:
                correct += 1
            else:
                incorrect += 1
        oresult += result
    regr = linear_model.LogisticRegression()
    regr.fit(dataset.xtable, dataset.ytable)
    y_test_scaled = testset.ytable
    y_pred_scaled = regr.predict(testset.xtable)
    y_pred_scaled = list(y_pred_scaled)
    y_pred,result = test_algorithm(testset,y_pred_scaled, y_test_scaled, s_min, s_max, mlr_test)
    print("Percentage of error over the MLR Cross-Validation: {} Test: {}. Correct: {} Incorrect: {}".format(oresult/nfold,result,correct,incorrect))

def main():
    execute_svm(dataset, testset, best_svm_params, y_test_scaled, s_min,s_max,svm_test, setname)
    execute_bp(dataset, testset, best_bp_params, y_test_scaled, s_min,s_max,bp_test, setname)
    execute_mlr(dataset, testset, y_test_scaled, s_min,s_max,mlr_test, setname)

if __name__ == "__main__":
    # To pass paramters
    parser = argparse.ArgumentParser(description='A test program.')
    # params_file: a text file which contains parameters for the execution
    parser.add_argument("params_file", type=str, nargs=1, help="Give the params_file")
    args = parser.parse_args()
    # output files
    bp_test = "_BP_test.csv"
    mlr_test = "_MLR_test.csv"
    svm_test = "_SVM_test.csv"
    # Initiate variable
    best_bp_params = best_svm_params = activation = epochs = alpha = n = alpha = trainset_size_perc = dataset_name = s_max = s_min = decision = None
    # read parameters
    dataset_name, trainset_size_perc, s_min, s_max, best_bp_params, best_svm_params, decision = read_params(args)
    # dataset name
    setname = dataset_name.split('-')[1].split('.')[0]
    # read dataset
    dataset, testset = read_dataset(dataset_name)
    y_test = np.copy(testset.ytable)
    # scale dataset
    scale_dataset(dataset,s_min,s_max)
    scale_dataset(testset,s_min,s_max)
    y_test_scaled = np.copy(testset.ytable)

    if decision == "main":
        main()
    elif decision == "cross_validation":
        cross_validation()
    else:
        print('Please put the right decision\n\t[main] to run the main algorithm using the params in file\n\t[run_all_params] to run all possible params from params.csv file\n\t[find_best_params] to find the best params to use after running all of them')