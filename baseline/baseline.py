import argparse

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

from keras.models import Sequential
from keras.layers import Dense, Flatten, ELU, Activation, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.initializers import glorot_uniform
from random import randrange
from keras.utils.np_utils import to_categorical

#from common import load_data, load_data_hog, train_and_save_results, L2Penalty

seed=10102016

def get_model_1(learning):
        
    image_shape = (64, 64, 1)

    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=image_shape, output_shape=image_shape))
    
    model.add(Convolution2D(
        5, 
        5, 
        kernel_initializer='random_uniform',
        bias_initializer='zeros',
        border_mode="valid"))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(7, kernel_initializer=glorot_uniform(seed), activation='softmax'))

    optimizer = Adam(lr=learning)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def get_model_2(learning):
        
    image_shape = (64, 64, 1)

    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=image_shape, output_shape=image_shape))
    
    model.add(Convolution2D(
        5, 
        5, 
        kernel_initializer='random_uniform',
        bias_initializer='zeros',
        border_mode="valid"))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))

    model.add(Convolution2D(
        10, 
        5, 
        kernel_initializer='random_uniform',
        bias_initializer='zeros',
        border_mode="valid"))
    model.add(MaxPooling2D())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(7, kernel_initializer=glorot_uniform(seed), activation='softmax'))

    optimizer = Adam(lr=learning)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def load_dataset(file_train, file_test, size=1.0):
    data_train = np.loadtxt(file_train, dtype='str', delimiter=',')
    data_test = np.loadtxt(file_test, dtype='str', delimiter=',')
    
    xtr, ytr = data_train[:,1], data_train[:,0].astype(int)
    xte, yte = data_test[:,1], data_test[:,0].astype(int)
    
    # use 15% of train data for testing
    xtr, xva, ytr, yva = train_test_split(xtr, ytr, test_size=0.30)

    if size >= 1.0 and size <=0:
        return xtr, ytr, xva, yva, xte, yte
    
    discard_size = 1.0 - size
    xtr, _, ytr, _ = train_test_split(xtr, ytr, test_size=discard_size)
    xva, _, yva, _ = train_test_split(xva, yva, test_size=discard_size)
    xte, _, yte, _ = train_test_split(xte, yte, test_size=discard_size)
    
    xtr, ytr = reshape_dataset(xtr, ytr.ravel())
    xva, yva = reshape_dataset(xva, yva.ravel())
    xte, yte = reshape_dataset(xte, yte.ravel())
    
    return xtr, ytr, xva, yva, xte, yte

def reshape_dataset(x, y):
    # make sure classes are between 0 and num_classes
    new_y = np.ones_like(y)*(-1)
    num_classes = np.unique(y)
    for idx, label in enumerate(num_classes):
        new_y[y == label] = idx
    
    y = to_categorical(new_y)
    
    return x, y

def load_clothes(size=1.0):
    return load_dataset("../data/clothes_train.txt", "../data/clothes_test.txt", size)
    
def load_faces(size=1.0):
    return load_dataset("../data/faces_train.txt", "../data/faces_test.txt", size)

def save_plot_metrics(log_file_name, history):
    keys = history.history.keys()

    f, ax = plt.subplots(len(keys), 1, figsize=(5, 22))

    for idx, k in enumerate(keys):
        ax[idx].plot(history.history[k])
        ax[idx].set_title("model " + k)
        ax[idx].set_ylabel(k)
        ax[idx].set_xlabel('epoch')
    
    f.savefig("{:s}.png".format(log_file_name), dpi=90)

def save_log_metrics(log_file_name, hyper, history):
    header = ""

    for key in hyper:
        header = header + ", " + key + ": " + str(hyper[key])

    header = header[2:]

    with open(log_file_name + ".txt", "w+") as log_file:
        log_file.write(header+"\n")
        
        keys = history.history.keys()
        head = ""
        
        c = 0
        for k in keys:
            if c == 0:
                l = len(history.history[k]) # number of epochs
                h = np.zeros(l)
            head = head + k + ","
            h = np.vstack((h, history.history[k]))
            c = c + 1

        head = head[:-1]
        head = head + "\n"
        log_file.write(head)

        h = h[1:,:]
        h = h.T

        for row in h:
            new_line = ""
            for value in row:
                new_line = new_line + "{:.8f},".format(value)
            new_line = new_line[:-1]
            new_line = new_line + "\n"
            log_file.write(new_line)

    log_file.close()

def generator(X, y, batch_size, augment=False):
    total_input = len(X)
    
    while True:
        features, targets = [], []
        i = 0
        while len(features) < batch_size:
            index = randrange(0, total_input)
            feats = X[index]
            labels = y[index]

            image = open_image(feats, augment)
                               
            features.append(image)
            targets.append(labels)
            
        yield (np.array(features), np.array(targets))

def getFeaturesTargets(X, y, augment=False):
    feats = []
    targets = []

    for feat, label in zip(X, y):
        image = open_image(feat, augment)
           
        feats.append(image)
        targets.append(label)

    return np.array(feats), np.array(targets)

def open_image(path, augment=True):
    image_path = "../data" + path[1:]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if augment == False:
        return image.reshape((64, 64, 1))

    # Rotation
    if randrange(3) == 0:
        angle = randrange(20, 45)
        image = rotate(image, angle)

    # Blur
    if randrange(3) == 0:
        image = cv2.blur(image, (5,5))

    # Skew
    if randrange(3) == 0:
        side = randrange(4)
        percentage = randrange(5, 15)
        rev = randrange(1)
        image = skew(image, side, percentage, rev == 0)

    # Shift
    if randrange(3) == 0:
        percentage = randrange(5)
        image = shift(image, percentage)
    
    return image.reshape((64, 64, 1))

def rotate(image, angle=45):
    h, w = image.shape
    cx = randrange((int)(0.375 * w), (int)(0.625 * w))
    cy = randrange((int)(0.375 * h), (int)(0.625 * h))

    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1)

    image = cv2.warpAffine(image, matrix, (w,h))

    return image

def skew(image, side=0, percentage=25, reverse=False):
    h, w = image.shape

    # side 0, 1, 2, 3
    # top, bottom, left, right
    if side == 0 or side == 1:
        side_length = w
    else:
        side_length = h
    
    # remove this amount of side in the beggining and the end
    sd = (int)(side_length * percentage / 100 / 2)

    if side == 0:
        pts1 = np.float32([[sd,0],[w-1-sd,0],[0,h-1],[w-1,h-1]])
    elif side == 1:
        pts1 = np.float32([[0,0],[w-1,0],[sd,h-1],[w-1-sd,h-1]])
    elif side == 2:
        pts1 = np.float32([[0,sd],[0,w-1],[0,h-1-sd],[w-1,h-1]])
    elif side == 3:
        pts1 = np.float32([[0,0],[w-1,sd],[0,h-1],[w-1,h-1-sd]])

    pts2 = np.float32([[0,0],[w-1, 0],[0, h-1],[w-1, h-1]])

    if reverse:
        m = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        m = cv2.getPerspectiveTransform(pts1, pts2)

    return cv2.warpPerspective(image, m, (w,h))

def shift(image, percentage=20, negative=False):
    h,w = image.shape

    y = (int)(h*percentage/100)
    x = (int)(w*percentage/100)

    if negative:
        x = x*-1
        y = y*-1

    m = np.float32([[1,0,x],[0,1,y]])

    return cv2.warpAffine(image, m, (w,h))

# arg dataset is 0 for clothes, 1 for faces
def train_model(model, hyper):
    learning_rate = hyper["learning_rate"]
    training_size = hyper["training_size"]
    batch_size = hyper["batch_size"]
    num_epochs = hyper["num_epochs"]
    dataset = hyper["dataset_type"]
    dataset_size = hyper["dataset_size"]
    data_augmentation = hyper["data_augmentation"]

    log_file_name = generate_log_file_name(hyper)

    if dataset == 1:
        xtr, ytr, xva, yva, xte, yte = load_clothes(dataset_size / 100.0)
    elif dataset_type == 2:
        xtr, ytr, xva, yva, xte, yte = load_faces(dataset_size / 100.0)
    else:
        raise NotImplementedError

    history = model.fit_generator(
        generator(xtr, ytr, batch_size, data_augmentation),
        samples_per_epoch = training_size,
        validation_data = getFeaturesTargets(xva, yva),
        nb_epoch = num_epochs
        )

    evalx, evaly = getFeaturesTargets(xte, yte)
    eval_ = model.evaluate(evalx, evaly)
    for val, key in zip(eval_, model.metrics_names):
        hyper[key] = val

    save_log_metrics(log_file_name, hyper, history)
    save_plot_metrics(log_file_name, history)

def generate_log_file_name(hyper):
    exp_name = hyper["exp_name"]
    model_type = hyper["model_type"]
    
    dataset_type = hyper["dataset_type"]
    if dataset_type == 1:
        dataset_name = "clothes"
    elif dataset_type == 2:
        dataset_name = "faces"
    else:
        raise NotImplementedError

    dataset_size = hyper["dataset_size"]
    return "{:s}_model_{:d}_{:s}_{:d}".format(exp_name, model_type, dataset_name, dataset_size)

def train_networks(exp_name, model_type, learning_rate, training_size, batch_size, num_epochs, dataset_size, dataset_type,
    data_augmentation):
    model = None

    if model_type == 1:
        model = get_model_1(learning_rate)
    elif model_type == 2:
        model = get_model_2(learning_rate)
    else:
        raise NotImplementedError

    hyper = OrderedDict()
    hyper["learning_rate"] = learning_rate
    hyper["training_size"] = training_size
    hyper["batch_size"] = batch_size
    hyper["num_epochs"] = num_epochs
    hyper["dataset_type"] = dataset_type
    hyper["dataset_size"] = dataset_size
    hyper["data_augmentation"] = data_augmentation
    hyper["model_type"] = model_type
    hyper["exp_name"] = exp_name

    train_model(model, hyper)

#########################################################################################
# from https://stackoverflow.com/a/43357954
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="CNN systems for coursework 3")
parser.add_argument('exp_name', type=str, help="Name of experiment")
parser.add_argument('model_type', type=int, help="Type of classifier")
parser.add_argument('-n', dest='num_epochs', type=int, default=100)
parser.add_argument('-l', dest='learning_rate', type=float, default=0.001)
parser.add_argument('-t', dest='training_size', type=int, default=2000)
parser.add_argument('-b', dest='batch_size', type=int, default=50)
parser.add_argument('-s', dest='dataset_size', type=int, default=100)
parser.add_argument('-d', dest='dataset_type', type=int, default=1)
parser.add_argument('-a', dest='data_augmentation', type=str2bool, default="false")

args = parser.parse_args()
exp_name = args.exp_name
model_type = args.model_type
num_epochs = args.num_epochs
learning_rate = args.learning_rate
training_size = args.training_size
batch_size = args.batch_size
dataset_size = args.dataset_size
dataset_type = args.dataset_type
data_augmentation = args.data_augmentation

train_networks(exp_name, model_type, learning_rate, training_size, batch_size, num_epochs, dataset_size, dataset_type,
    data_augmentation)