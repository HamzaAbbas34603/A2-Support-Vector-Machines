def BP_params():
    activations = ['tanh','logistic','relu']
    epochs = [1000,5000] #number of epochs
    n = [0.02,0.05,0.1] # learning rate [0.01 - 0.2]
    alpha = [0.5,0.9] # momentum [0.1 - 0.9]
    L = [2,3,4,5,6] #number of layers
    f = open('BP_params.csv', 'w', encoding='utf-8')
    f.write('Id,Activation,Epochs,Learning,Momentum,Layers\n')
    params = []
    cpt = 0
    for act in activations:
        for ep in epochs:
            for r in n:
                for m in alpha:
                    for l in L:
                        f.write('{},{},{},{},{},{}\n'.format(cpt,act,ep,r,m,l))
                        cpt += 1
                        params.append([act,ep,r,m,l])
    f.close()
    return params

def SVM_params():
    kernel = ['sigmoid','rbf','poly']
    C = [1,5,10,20,50]
    f = open('SVM_params.csv', 'w', encoding='utf-8')
    f.write('Id,Kernel,C\n')
    params = []
    cpt = 0
    for act in kernel:
            for r in C:
                params.append([act,r])
                f.write('{},{},{}\n'.format(cpt,act,r))
                cpt += 1
    f.close()
    return params