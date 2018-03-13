from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import glob
import time

def load_dataset(file_train, file_test, size=1.0):
	# loadtxt gives results with that b' char
    # data_train = np.loadtxt(file_train, dtype='str', delimiter=',')
    # data_test = np.loadtxt(file_test, dtype='str', delimiter=',')

    data_train = np.genfromtxt(file_train, dtype='str', delimiter=',')
    data_test = np.genfromtxt(file_test, dtype='str', delimiter=',')

    xtr, ytr = data_train[:,1], data_train[:,0].astype(int)
    xte, yte = data_test[:,1], data_test[:,0].astype(int)
    
    return xtr, ytr, xte, yte

def extract_features(model, img_path):
	img = image.load_img(img_path, target_size=(64, 64))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	features = model.predict(x)

	# Get only the max values of every 7x7 kernel
	features = features.ravel()
	return features

def get_features_from_dataset(model, paths, labels, records_file):
	new_feats = np.zeros(2048)

	with open(records_file, "a") as myfile:
		for path, label in zip(paths, labels):
			image_path = "../data" + path[1:]
			print(image_path)
			c_feats = extract_features(model, image_path)
			xy = np.hstack((label, c_feats))


			# from https://stackoverflow.com/a/13861407
			#generate an array with strings
			xy_arrstr = np.char.mod('%.2f', xy)
			# #combine to a string
			xy_str = ",".join(xy_arrstr)
			myfile.write(xy_str + "\n")

			#new_feats = np.vstack((new_feats, extract_features(model, image_path)))


	#return new_feats[1:,:] # No the first zero row


def process_dataset(dataset_name):
	model = VGG16(weights='imagenet', include_top=False)
	
	path_to_test = "../data/{:s}_test.txt".format(dataset_name)
	path_to_train = "../data/{:s}_train.txt".format(dataset_name)
	

	try:
		xtr, ytr, xte, yte = load_dataset(path_to_train, path_to_test)
	except Exception as e:
		raise e

	path_feats_to_test = "../data/{:s}_2048_test.txt".format(dataset_name)
	path_feats_to_train = "../data/{:s}_2048_train.txt".format(dataset_name)

	# xtr_feats = get_features_from_dataset(model, xtr, ytr, path_feats_to_train)
	# xte_feats = get_features_from_dataset(model, xte, yte, path_feats_to_test)
	
	s = time.time()
	get_features_from_dataset(model, xtr, ytr, path_feats_to_train)
	get_features_from_dataset(model, xte, yte, path_feats_to_test)
	e = time.time()

	print("total time: {:.2f}".format(e-s))
	# xy_tr = np.hstack((ytr, xtr_feats))
	# xy_te = np.hstack((yte, xte_feats))

	#np.savetxt("{:s}_train_512.txt".format(dataset_name), xy_tr, delimiter=',')
	#np.savetxt("{:s}_test_512.txt".format(dataset_name), xy_te, delimiter=',')


process_dataset("faces")
process_dataset("clothes")