#!/usr/bin/bash
###################################################
# 3 fully connected layers on clothes
###################################################

python deep_features_models.py c_0 1 -n 20 -d 1 -s 100
python deep_features_models.py c_0 1 -n 20 -d 1 -s 75
python deep_features_models.py c_0 1 -n 20 -d 1 -s 50
python deep_features_models.py c_0 1 -n 20 -d 1 -s 25
python deep_features_models.py c_0 1 -n 20 -d 1 -s 10
python deep_features_models.py c_0 1 -n 20 -d 1 -s 1

###################################################
# 4 fully connected layers on clothes
###################################################

python deep_features_models.py c_1 2 -n 20 -d 1 -s 100
python deep_features_models.py c_1 2 -n 20 -d 1 -s 75
python deep_features_models.py c_1 2 -n 20 -d 1 -s 50
python deep_features_models.py c_1 2 -n 20 -d 1 -s 25
python deep_features_models.py c_1 2 -n 20 -d 1 -s 10
python deep_features_models.py c_1 2 -n 20 -d 1 -s 1

###################################################
# 5 fully connected layers on clothes
###################################################

python deep_features_models.py c_2 3 -n 20 -d 1 -s 100
python deep_features_models.py c_2 3 -n 20 -d 1 -s 75
python deep_features_models.py c_2 3 -n 20 -d 1 -s 50
python deep_features_models.py c_2 3 -n 20 -d 1 -s 25
python deep_features_models.py c_2 3 -n 20 -d 1 -s 10
python deep_features_models.py c_2 3 -n 20 -d 1 -s 1

###################################################
# 3 fully connected layers on faces
###################################################

python deep_features_models.py f_0 1 -n 20 -d 2 -s 100
python deep_features_models.py f_0 1 -n 20 -d 2 -s 75
python deep_features_models.py f_0 1 -n 20 -d 2 -s 50
python deep_features_models.py f_0 1 -n 20 -d 2 -s 25
python deep_features_models.py f_0 1 -n 20 -d 2 -s 10
python deep_features_models.py f_0 1 -n 20 -d 2 -s 1

###################################################
# 4 fully connected layers on faces
###################################################

python deep_features_models.py f_1 2 -n 20 -d 2 -s 100
python deep_features_models.py f_1 2 -n 20 -d 2 -s 75
python deep_features_models.py f_1 2 -n 20 -d 2 -s 50
python deep_features_models.py f_1 2 -n 20 -d 2 -s 25
python deep_features_models.py f_1 2 -n 20 -d 2 -s 10
python deep_features_models.py f_1 2 -n 20 -d 2 -s 1

###################################################
# 5 fully connected layers on faces
###################################################

python deep_features_models.py f_2 3 -n 20 -d 2 -s 100
python deep_features_models.py f_2 3 -n 20 -d 2 -s 75
python deep_features_models.py f_2 3 -n 20 -d 2 -s 50
python deep_features_models.py f_2 3 -n 20 -d 2 -s 25
python deep_features_models.py f_2 3 -n 20 -d 2 -s 10
python deep_features_models.py f_2 3 -n 20 -d 2 -s 1