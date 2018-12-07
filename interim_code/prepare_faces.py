import numpy as np
import cv2
import glob
from shutil import copyfile

def prepare():

    expressions = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    max_samples_test = 900
    max_samples_train = 6000

    data_train = np.zeros(4097)
    data_test = np.zeros(4097)

    records_train = np.array(["-1", "fakepath"]).astype(str)
    records_test = np.array(["-1", "fakepath"]).astype(str)

    for label, exp in enumerate(expressions):
        nsamples_train = 0
        nsamples_test = 0
        
        for fname in glob.glob("./faces_cleaned/{:s}/*".format(exp)):

            if nsamples_test >= max_samples_test and nsamples_train >= max_samples_train:
                break

            elif nsamples_train < max_samples_train:
                new_fname = "./faces/train/{:d}/{:d}.jpg".format(label, nsamples_train)
                records_train = np.vstack((records_train, [label, new_fname]))
                nsamples_train = nsamples_train + 1

            elif nsamples_test < max_samples_test:
                new_fname = "./faces/test/{:d}/{:d}.jpg".format(label, nsamples_test)
                records_test = np.vstack((records_test, [label, new_fname]))
                nsamples_test = nsamples_test + 1

            copyfile(fname, new_fname)

    records_train = records_train[1:,:]
    records_test = records_test[1:,:]

    np.savetxt("faces_train.txt", records_train, delimiter=',', fmt='%s')
    np.savetxt("faces_test.txt", records_test, delimiter=',', fmt='%s')


prepare()




