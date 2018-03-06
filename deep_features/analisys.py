import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from collections import OrderedDict

def load_results(file_name):
    
    # dataset, size, model, val acc, val err
    results= np.array(["Dataset", "Size", "Model", "Accuracy", "Error"])
        
    for size in [100, 75, 50, 25, 10, 1]:
        for exp_number in [0, 1, 2]:
            model = exp_number + 1
            file_face = "f_{:d}_model_{:d}_faces_{:d}.txt".format(exp_number, model, size)
            file_clot = "c_{:d}_model_{:d}_clothes_{:d}.txt".format(exp_number, model, size)
            
            hyper_face = get_hyper(file_face)
            hyper_clot = get_hyper(file_clot)
            
            top_line = ""
            face_line = ""
            clot_line = ""
            
            row_face = np.array([])
            row_clot = np.array([])
            for key, key2 in zip(hyper_face, hyper_clot):
                
                if key not in ["dataset_type", "vAcc", "vErr", "model_type", "dataset_size"]:
                    continue
                
                row_face = np.hstack((row_face, (str)(hyper_face[key])))
                row_clot = np.hstack((row_clot, (str)(hyper_clot[key])))
                                
            results = np.vstack((results, row_face, row_clot))
    
    np.savetxt(file_name, results[:,:],  fmt='%s', delimiter=',')

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

load_results("deep_features_analysis.txt")