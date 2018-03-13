#!/usr/bin/bash
###################################################
# 3 fully connected layers on clothes
###################################################

#python deep_features_models.py baseline 1 -n 20 -d 1 -s 100
models='1 2 3'
activations='elu relu sigmoid'
learning_rates='0.1 0.01 0.001 0.0001'
datasets='1 2'

for  dataset in $datasets
do
	for model in $models
	do
		for activation in $activations
		do 
			for learning_rate in $learning_rates
			do
				echo Dataset: $dataset 
				echo Model: $model
				echo Activation: $activation 
				echo Learning rate: $learning_rate 

				python baseline_deep_features_models.py baseline $model -d $dataset -l $learning_rate -a $activation -n 20 -s 100 
			done
		done
	done
done


echo All done

