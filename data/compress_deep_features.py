import numpy as np


def load_dataset(file_train, file_test):
    data_train = np.genfromtxt(file_train, dtype='float', delimiter=',')
    data_test = np.genfromtxt(file_test, dtype='float', delimiter=',')

    xtr, ytr = data_train[:,1:], data_train[:,0].astype(int)
    xte, yte = data_test[:,1:], data_test[:,0].astype(int)
    
    return xtr, ytr, xte, yte

def save(x, y, name):
    file = open(name, "wb")
    np.savez(file, inputs=x, targets=y)

def compress(dataset_name):
    path_to_test = "./{:s}_512_test.txt".format(dataset_name)
    path_to_train = "./{:s}_512_train.txt".format(dataset_name)

    xtr, ytr, xte, yte = load_dataset(path_to_train, path_to_test)
    print(xtr.shape)
    print(ytr.shape)
    print(xte.shape)
    print(yte.shape)

    path_to_npz_test = "./{:s}_512_test.npz".format(dataset_name)
    path_to_npz_train = "./{:s}_512_train.npz".format(dataset_name)
    
    save(xtr, ytr, path_to_npz_train)
    save(xte, yte, path_to_npz_test)

compress("faces")
compress("clothes")