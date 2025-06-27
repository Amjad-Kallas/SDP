import numpy as numpy

def vrow(input):
    return input.reshape((1, input.size))

def vcol(input):
    return input.reshape((input.size, 1))

def load_data(file_path):
    # Load the data from CSV file
    data = numpy.loadtxt(file_path, delimiter=',', skiprows=1)
    
    # Split features (D) and labels (L)
    D = data[:, :-1] 
    L = data[:, -1]  
    
    return D.T, L.T # I need to transpose the data because the data is in the wrong shape

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)
