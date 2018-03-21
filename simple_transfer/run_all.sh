#!/usr/bin/bash

# transfer from clothes to expressions, load expressions dataset
python simple_transfer.py simple_transfer 1 -n 10 -d 2 -s 100 -w "./1_model_1_clothes_100.hdf"
python simple_transfer.py simple_transfer 1 -n 10 -d 2 -s 1 -w "./1_model_1_clothes_100.hdf"

# transfer from expressions to clothes, load clothes dataset
python simple_transfer.py simple_transfer 1 -n 10 -d 1 -s 100 -w "./3_model_1_faces_100.hdf"
python simple_transfer.py simple_transfer 1 -n 10 -d 1 -s 1 -w "./3_model_1_faces_100.hdf"

# DATA AUGMENTATION

# transfer from clothes to expressions, load expressions dataset
python simple_transfer.py simple_transfer_da 1 -n 10 -d 2 -a y -s 100 -w "./1_model_1_clothes_100.hdf"
python simple_transfer.py simple_transfer_da 1 -n 10 -d 2 -a y -s 1 -w "./1_model_1_clothes_100.hdf"

# transfer from expressions to clothes, load clothes dataset
python simple_transfer.py simple_transfer_da 1 -n 10 -d 1 -a y -s 100 -w "./3_model_1_faces_100.hdf"
python simple_transfer.py simple_transfer_da 1 -n 10 -d 1 -a y -s 1 -w "./3_model_1_faces_100.hdf"