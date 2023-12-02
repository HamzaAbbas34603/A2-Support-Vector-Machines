"""
The file contains a list of functions that are util to the main program
    read_dataset: read data from dataset files, put features in xtable, classes in ytable
    print_dataset: display dataset in output
    scale_dataset: apply normalization and scaling for the dataset, make all values between 0.1 and 0.9
    descale_dataset
    descale_value: descale only class values
    plot_results: make a scatter plot to show the classes
    order_best_params
    read_params: read paramters that are in params.txt file to execute the program
"""
from sklearn.decomposition import PCA
import numpy as np
import timeit
import math
import random
import matplotlib.pyplot as plt

class Dataset:
    nf: int #number of features
    no: int #number of ouputs
    ns: int #number of samples
    xtable: list = [] #array of arrays of features
    ytable: list = [] # array of arrays of outputs
    xmin: list #array with minimum x for each feature
    ymin: int #array with minimum x for each output
    xmax: list #array with maximum x for each feature
    ymax: int #array with maximum x for each output
    def __init__(self,nf,no,ns,xtable,ytable):
        self.nf = nf
        self.no = no
        self.ns = ns
        self.xtable = xtable
        self.ytable = ytable
        self.xmin = []
        self.xmax = []
        self.ymin = 0
        self.ymax = 0

# function to read dataaset
def read_dataset(file_name):
    if trainset_size_perc == 100 and 'ring' in file_name:
        xtable = []
        ytable = []
        with open(file_name) as f:
            _,nf,no = map(int,f.readline().split(' '))
            data = [list(map(float,d.replace('\n','').split('\t'))) for d in f.readlines()]
            ns = len(data)
            for row in data:
                xtable.append(row[:nf])
                ytable.append(row[nf])
        dataset = Dataset(nf,no,ns,xtable,ytable)
        xtable = []
        ytable = []
        with open('A2-ring-test.txt') as f:
            _,nf,no = map(int,f.readline().split(' '))
            data = [list(map(float,d.replace('\n','').split('\t'))) for d in f.readlines()]
            ns = len(data)
            for row in data:
                xtable.append(row[:nf])
                ytable.append(row[nf])
        testset = Dataset(nf,no,ns,xtable,ytable)
    else:
        xtable = []
        ytable = []
        xtable2 = []
        ytable2 = []
        with open(file_name) as f:
            _,nf,no = map(int,f.readline().split(' '))
            data = [list(map(float,d.replace('\n','').split('\t'))) for d in f.readlines()]
            ns = len(data) * trainset_size_perc // 100
            ns2 = len(data) - len(data) * trainset_size_perc // 100
            for row in data[:ns]:
                xtable.append(row[:nf])
                ytable.append(row[nf])
            for row in data[ns:]:
                xtable2.append(row[:nf])
                ytable2.append(row[nf])
        dataset = Dataset(nf,no,ns,xtable,ytable)
        testset = Dataset(nf,no,ns2,xtable2,ytable2)
    return dataset, testset
        
# display dataset (not used in the main algorithm)
def print_dataset(dataset):
    for m in range(0,dataset.ns):
        for n in range(0,dataset.nf):
            print("{}".format(dataset.xtable[m][n]),end='\t')
        for n in range(0,dataset.no):
            print("{}".format(dataset.ytable[m]),end='\t')
        print()

# function to scale the dataset
def scale_dataset(dataset,s_min,s_max):
    dataset.xmin = np.zeros(dataset.nf)
    dataset.xmax = np.zeros(dataset.nf)
    for n in range(0,dataset.nf):
        max = float('-inf')
        min = float('inf')
        for m in range(0,dataset.ns):
            if dataset.xtable[m][n] > max:
                max = dataset.xtable[m][n]
            if dataset.xtable[m][n] < min:
                min = dataset.xtable[m][n]
        dataset.xmin[n] = min
        dataset.xmax[n] = max
        if min == max:
            min = 0
            max = 1
        for m in range(0, dataset.ns):
            dataset.xtable[m][n] = s_min + (s_max - s_min)/(max - min)*(dataset.xtable[m][n] - min)

    max = float('-inf')
    min = float('inf')
    for m in range(0,dataset.ns):
        if dataset.ytable[m] > max:
            max = dataset.ytable[m]
        if dataset.ytable[m] < min:
            min = dataset.ytable[m]
    dataset.ymin = min
    dataset.ymax = max
    for m in range(0,dataset.ns):
        dataset.ytable[m] = s_min + (s_max - s_min)/(max - min)*(dataset.ytable[m] - min)
    return dataset

# function to descale dataset (return normal values)
def descale_dataset(dataset,s_min,s_max):
    for n in range(0,dataset.nf):
        min = dataset.xmin[n]
        max = dataset.xmax[n]
        for m in range(0,dataset.ns):
            dataset.xtable[m][n] = min + (max - min)/(s_max - s_min)*(dataset.xtable[m][n] - s_min)
    min = dataset.ymin
    max = dataset.ymax
    for m in range(0,dataset.ns):
        dataset.ytable[m] = min + (max - min)/(s_max - s_min)*(dataset.ytable[m] - s_min)
    return dataset

# function to descale a value y
def descale_y_value(dataset,value,s_min,s_max):
    y_min = dataset.ymin
    y_max = dataset.ymax
    return ( y_min + (y_max - y_min)/(s_max - s_min)*(value - s_min) )

# function to display results scatter plot using pca
def plot_results(dataset,y_pred):
    if dataset.nf > 2:
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(dataset.xtable)
        xtable = np.transpose(principalComponents)
    else:
        xtable = np.transpose(dataset.xtable)
    y_pred = ['blue' if x==0 else 'red' for x in y_pred]
    plt.scatter(xtable[0],xtable[1],c=y_pred)
    plt.show()

# function to order and find best parametrs based on running the algorithms
def order_best_params(setname):
    f = open('params.csv',encoding='utf-8')
    errors = []
    for item in f.readlines()[1:]:
        try:
            item = item.replace('\n','').split(',')
            h = open('outputs-{}/errors_{}_0.csv'.format(setname,item[0]))
            last = h.readlines()[-1]
            qt = float(last.replace('\n','').split(',')[1])
            qv = float(last.replace('\n','').split(',')[2])
            errors.append({'qt':qt,'qv':qv,'id':item[0],'params':item[1:]})
            h.close()
        except FileNotFoundError:
            pass
    errors = sorted(errors,key=lambda x:x['qv'])
    for err in errors[:10]:
        qv = 0
        qt = 0
        for i in range(0,10):
            h = open('outputs-{}/errors__train_{}_{}.csv'.format(setname,err.get('id'),i))
            last = h.readlines()[-1]
            qt += float(last.replace('\n','').split(',')[1])
            qv += float(last.replace('\n','').split(',')[2])
            h.close()
        print("Training ID: {}\tAttempt {}\tQuadratic error train set: {}\tQuadratic error validation set: {}\t Params: {}".format(err.get('id'),i, qt/10, qv/10,err.get('params')))

# function to read exectuion parameters from a text file
def read_params(args):
    global activation, epochs, alpha, n, alpha, nn, trainset_size_perc, dataset_name, s_max, s_min, decision, best_bp_params, best_svm_params
    f = open(args.params_file[0])
    params = f.readlines()
    params = [param.replace('\n','') for param in params]
    dataset_name = params[0]
    trainset_size_perc = int(params[1])
    s_min, s_max = [float(i) for i in params[2].split(' ')]
    best_bp_params = params[3].split(' ')
    best_svm_params = params[4].split(' ')
    decision = params[5]
    return dataset_name, trainset_size_perc, s_min, s_max, best_bp_params, best_svm_params, decision

# function to give predicted values to test the training
def test_algorithm(dataset,y_pred, y_scaled,s_min,s_max, output_file,setname):
    outFile = open('outputs-{}/{}'.format(setname,output_file), "w")
    aux1 = aux2 = 0
    output = []
    for i in range(len(y_scaled)):
        y = descale_y_value(dataset,y_pred[i], s_min, s_max)
        z = descale_y_value(dataset,y_scaled[i], s_min, s_max)
        output.append(y)
        outFile.write("{}, {}, {}\n".format(z, y, abs(z - y)))
        aux1 += (1 if z!=y else 0)
        aux2 += 1
    outFile.close()
    return output,100*aux1/aux2