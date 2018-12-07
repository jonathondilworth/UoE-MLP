#!/usr/bin/bash
###################################################
# 3 fully connected layers on clothes
###################################################

#python deep_features_models.py baseline 1 -n 20 -d 1 -s 100
models='1 2 3'
activations='elu sigmoid'
learning_rates='0.01 0.001 0.0001'
datasets='1 2'
sizes='100 10 1 0.1'

for  dataset in $datasets
do
	for model in $models
	do
		for activation in $activations
		do 
			for learning_rate in $learning_rates
			do
				for size in $sizes
				do
					echo Dataset: $dataset 
					echo Model: $model
					echo Activation: $activation 
					echo Learning rate: $learning_rate 
					echo Dataset size: $size

					python baseline_deep_features_models.py baseline $model -d $dataset -l $learning_rate -a $activation -s $size -n 20
				done
			done
		done
	done
done


echo All done

