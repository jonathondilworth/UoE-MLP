import numpy as np
import cv2

def load_clothes():
    train = np.loadtxt("./clothes/fashion-mnist_train.csv", skiprows=1, dtype=int, delimiter=',')
    test = np.loadtxt("./clothes/fashion-mnist_test.csv", skiprows=1, dtype=int, delimiter=',')
    
    # testing of code in notebook
    # train = train[:100, :]
    # test = test[:100, :]

    xtr, ytr = split_xy(train)
    xte, yte = split_xy(test)
    
    return xtr, ytr, xte, yte


def split_xy(data):
    return data[:,1:], data[:,0]

def resize_clothes(x):
    new_set = np.zeros(4096)
    
    for sample in x:
        sample = sample.astype(float)
        resized = cv2.resize(sample.reshape(28, 28), (64, 64))
        new_set = np.vstack((new_set, resized.ravel()))
        
    return new_set[1:]

def save(x, y, name):
    file = open(name, "wb")
    np.savez(file, inputs=x, targets=y)

def filter(data, valid_labels):
    valid_data = np.zeros(data.shape[1])
    for sample in data:
        if sample[0] in valid_labels:
            valid_data = np.vstack((valid_data, sample))


    return valid_data[1:,:]

def prepare():
    c_xtr, c_ytr, c_xte, c_yte = load_clothes()
    c_xtr = resize_clothes(c_xtr)
    c_xte = resize_clothes(c_xte)

    # Get only seven classes discard 7, 6, 4
    c_ytr = c_ytr.reshape((-1, 1))
    c_yte = c_yte.reshape((-1, 1))

    train = np.hstack((c_ytr, c_xtr))
    test = np.hstack((c_yte, c_xte))

    valid_labels = [0,1,2,3,5,8,9]

    train = filter(train, valid_labels)
    test = filter(test, valid_labels)

    xtr, ytr = split_xy(train)
    xte, yte = split_xy(test)

    save(xtr, ytr, "clothes_train.npz")
    save(xte, yte, "clothes_test.npz")

prepare()