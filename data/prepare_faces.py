import numpy as np
import cv2
import glob

def load_data():

    expressions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    max_samples_test = 900
    max_samples_train = 6000

    data_train = np.zeros(4097)
    data_test = np.zeros(4097)

    for label, exp in enumerate(expressions):
        nsamples_train = 0
        nsamples_test = 0
        
        for fname in glob.glob("./cleaned/{:s}/*".format(exp)):
            image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).ravel()
            sample = np.hstack(([label], image))

            if nsamples_test >= max_samples_test and  nsamples_train >= max_samples_train:
                break

            elif nsamples_train < max_samples_train:
                data_train = np.vstack((data_train, sample))
                nsamples_train = nsamples_train + 1

            elif nsamples_test < max_samples_test:
                data_test = np.vstack((data_test, sample))
                nsamples_test = nsamples_test + 1

    return data_train[1:,:], data_test[1:,:]

def split_xy(data):
    return data[:,1:], data[:,0]

def save(x, y, name):
    file = open(name, "wb")
    np.savez(file, inputs=x, targets=y)

def prepare():
    train, test = load_data()
    xtr, ytr = split_xy(train)
    xte, yte = split_xy(test)

    save(xtr, ytr, "faces_train.npz")
    save(xte, yte, "faces_test.npz")

prepare()




