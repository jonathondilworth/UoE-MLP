import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from collections import OrderedDict

def load_results(log_name):
    
    # dataset, size, model, val acc, val err
    results= np.array(["Learning rate", "Dataset", "Model", "Activation", "Accuracy", "Error"])

    for dataset in ["faces", "clothes"]:
        for model in [1, 2, 3]:
            for activation in ["relu", "elu", "sigmoid"]:
                for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
                    lr = str(learning_rate).replace(".", "")
                    file_name = "./baseline_experiments/baseline_M_{:d}_A_{:s}_L_{:s}_D_{:s}.txt".format(model, activation, lr, dataset)
                    hyper = get_hyper(file_name)
                    row = np.array([])
                    for key in hyper:
                        if key not in ["dataset_type", "model_type", "activation", "learning_rate", "vAcc", "vErr"]:
                            continue

                        row = np.hstack((row, (str)(hyper[key])))
                    results = np.vstack((results, row))

    print(results)
    np.savetxt(log_name, results,  fmt='%s', delimiter=',')

def get_hyper(file_name):
    with open(file_name, 'r') as f:
        hyper_line = f.readline()
    
    key_values = hyper_line.split(",")
    
    hyper = OrderedDict()
    
    for kv in key_values:
        key_val = kv.split(":")
        key = key_val[0].strip()
        try:
            val = (float)(key_val[1])
        except:
            val = (str)(key_val[1])
        
        hyper[key] = val
    
    # get val accuracy and error in last epoch
    epochs = np.genfromtxt(file_name, np.float32, delimiter=",", skip_header=2).astype(float)
    val_acc = epochs[-1,1]
    val_err = epochs[-1,0]
    
    hyper["vAcc"] = val_acc
    hyper["vErr"] = val_err
    if hyper['dataset_type'] == 1:
        hyper["dataset_type"] = "clothes"
        
    if hyper["dataset_type"] == 2:
        hyper["dataset_type"] = "faces"
        
    return hyper

load_results("baseline_deep_features_analysis.txt")