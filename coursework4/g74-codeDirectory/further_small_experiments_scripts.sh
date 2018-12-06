#!/usr/bin/bash
###################################################
# 3 fully connected layers on clothes
###################################################

#python deep_features_models.py baseline 1 -n 20 -d 1 -s 100
# Best result for 1% for clothes
python baseline_deep_features_further_experiments.py baseline 1 -d 1 -l 0.001 -a sigmoid -s 1 -n 200
# Best result for 1% for expressions
python baseline_deep_features_further_experiments.py baseline 2 -d 2 -l 0.001 -a elu -s 1 -n 200
# Best result for 0.1% for clothes
python baseline_deep_features_further_experiments.py baseline 2 -d 1 -l 0.01 -a sigmoid -s 0.1 -n 200
# Best result for 0.1% for expressions
python baseline_deep_features_further_experiments.py baseline 3 -d 2 -l 0.001 -a elu -s 0.1 -n 200
