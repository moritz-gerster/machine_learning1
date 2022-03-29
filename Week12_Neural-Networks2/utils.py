import numpy, numpy.random

import sklearn,sklearn.datasets

def breast_cancer():
    
    D = sklearn.datasets.load_breast_cancer()
    X = D['data']
    T = D['target'] * 2 - 1

    # Partition the data
    N = len(X)
    perm = numpy.random.mtrand.RandomState(0).permutation(N)
    Xtrain,Xtest = X[perm[:N//2]],X[perm[N//2:]]
    Ttrain,Ttest = T[perm[:N//2]],T[perm[N//2:]]
    
    Ttrain[::25]*= -1 # mislabel 4% of training data

    # Normalize input data
    m,s = Xtrain.mean(axis=0),Xtrain.std(axis=0)+1e-9
    for x in Xtrain,Xtest: x -= m; x /= s

    return Xtrain,Ttrain,Xtest,Ttest

