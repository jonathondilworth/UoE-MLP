import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from collections import OrderedDict


def load_results(file_name):
    # dataset, size, data_aug, val acc
    results= np.array(["", "", "", ""])
        
    for size in [100, 75, 50, 25, 10, 1]:
        for exp_clot, exp_face in zip([0,1], [2,3]):        
            file_face = "{:d}_model_1_faces_{:d}.txt".format(exp_face, size)
            file_clot = "{:d}_model_1_clothes_{:d}.txt".format(exp_clot, size)
            
            hyper_face = get_hyper(file_face)
            hyper_clot = get_hyper(file_clot)
            
            top_line = ""
            face_line = ""
            clot_line = ""
            
            row_face = np.array([])
            row_clot = np.array([])
            for key, key2 in zip(hyper_face, hyper_clot):
                
                if key not in ["dataset_type", "vAcc", "data_augmentation", "dataset_size"]:
                    continue
                
                row_face = np.hstack((row_face, (str)(hyper_face[key])))
                row_clot = np.hstack((row_clot, (str)(hyper_clot[key])))
                                
            results = np.vstack((results, row_face, row_clot))
    
    df = pd.DataFrame(results[1:,:])
    df.columns = ["Name", "Size", "Augmentation", "Accuracy"]

    # pd.DataFrame.to_csv(file_name, fmt='%s')
    np.savetxt(file_name + ".txt", df.values, fmt='%s')

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
    
        
    # get val accuracy in last epoch
    
    epochs = np.genfromtxt(file_name, np.float32, delimiter=",", skip_header=2).astype(float)
    val_acc = epochs[-1,1]
    
    hyper["vAcc"] = val_acc
    if hyper['dataset_type'] == 1:
        hyper["dataset_type"] = "clothes"
        
    if hyper["dataset_type"] == 2:
        hyper["dataset_type"] = "faces"
        
    return hyper

parser = argparse.ArgumentParser(description="cw 4")
parser.add_argument('file_name', type=str, help="Name of analysis")
args = parser.parse_args()
file_name = args.file_name

load_results(file_name)