import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from collections import OrderedDict

def load_results(log_name):
    
    # dataset, size, model, val acc, val err
    results= np.array(["Learning rate", "Dataset", "Dataset size", "Model", "Activation", "Accuracy", "Error"])

    for dataset in ["faces", "clothes"]:
        for model in [1, 2, 3]:
            for activation in ["elu", "sigmoid"]:
                for learning_rate in [0.01, 0.001, 0.0001]:
                    for dataset_size in [100.0, 10.0, 1.0, 0.1]:
                        try:
                            lr = str(learning_rate).replace(".", "")
                            ds_size = str(dataset_size).replace(".", "")
                            file_name = "./baseline_experiments/baseline_M_{:d}_A_{:s}_L_{:s}_D_{:s}_S_{:s}.txt".format(model, activation, lr, dataset, ds_size)
                            hyper = get_hyper(file_name)
                            row = np.array([])
                            for key in hyper:
                                if key not in ["dataset_size", "dataset_type", "model_type", "activation", "learning_rate", "vAcc", "vErr"]:
                                    continue

                                row = np.hstack((row, (str)(hyper[key])))
                            results = np.vstack((results, row))
                        except:
                            print("Undefined experiment")
                            print("dataset: {:s}, model {:d}, activation {:s}, learning rate {:.4f}, size {:.1f}". format(
                                dataset, model, activation, learning_rate, dataset_size))


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
            val = val.replace('\n', '')
        
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
        hyper["dataset_type"] = "expressions"
        
    return hyper

load_results("baseline_deep_features_analysis.txt")