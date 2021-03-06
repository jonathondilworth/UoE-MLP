import argparse

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras import regularizers
from keras.utils.np_utils import to_categorical
from random import randrange

seed=10102016

def get_model_1(learning_rate, activ="relu", num_feats=2048):
    model = Sequential()
    model.add(Activation('relu', batch_input_shape=(None, num_feats)))
    
    model.add(
        Dense(
            100, 
            activation=activ,
            init='uniform',
            kernel_regularizer=regularizers.l2(0.01)
        )
    )
    
    model.add(
        Dense(
            7, 
            kernel_initializer=glorot_uniform(seed), 
            activation='softmax'
        )
    )

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

    return model

def get_model_2(learning_rate, activ="relu", num_feats=2048):
    model = Sequential()
    model.add(Activation('relu', batch_input_shape=(None, num_feats)))
    
    model.add(
        Dense(
            100, 
            activation=activ,
            init='uniform',
            kernel_regularizer=regularizers.l2(0.01)
        )
    )

    model.add(
        Dense(
            50, 
            activation=activ,
            init='uniform'
        )
    )
    
    model.add(
        Dense(
            7, 
            kernel_initializer=glorot_uniform(seed), 
            activation='softmax'
        )
    )

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

    return model

def get_model_3(learning_rate, activ="relu", num_feats=2048):
    model = Sequential()
    model.add(Activation('relu', batch_input_shape=(None, num_feats)))
    
    model.add(
        Dense(
            100, 
            activation=activ,
            init='uniform',
            kernel_regularizer=regularizers.l2(0.01)
        )
    )

    model.add(
        Dense(
            50, 
            activation=activ,
            init='uniform'
        )
    )

    model.add(
        Dense(
            25, 
            activation=activ,
            init='uniform'
        )
    )
    
    model.add(
        Dense(
            7, 
            kernel_initializer=glorot_uniform(seed), 
            activation='softmax'
        )
    )

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

    return model

def load_dataset(file_train, file_test, size=1.0):

    data_train = np.load(file_train)
    data_val = np.load(file_test)

    xtr, ytr = data_train['inputs'].astype(np.float32), data_train['targets'].astype(int)
    xva, yva = data_val['inputs'].astype(np.float32), data_val['targets'].astype(int)

    # use 30% of train data for testing
    xtr, xte, ytr, yte = train_test_split(xtr, ytr, test_size=0.30)

    if size >= 1.0 and size <=0:
        return xtr, ytr, xva, yva, xte, yte
    
    discard_size = 1.0 - size
    xtr, _, ytr, _ = train_test_split(xtr, ytr, test_size=discard_size)
    # Keep the size of the validation constant
    # xva, _, yva, _ = train_test_split(xva, yva, test_size=discard_size) 
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
    return load_dataset("../data/deep_features/clothes_2048_train.npz", "../data/deep_features/clothes_2048_test.npz", size)
    
def load_faces(size=1.0):
    return load_dataset("../data/deep_features/faces_2048_train.npz", "../data/deep_features/faces_2048_test.npz", size)

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

# arg dataset is 0 for clothes, 1 for faces
def train_model(model, hyper):
    training_size = hyper["training_size"]
    batch_size = hyper["batch_size"]
    num_epochs = hyper["num_epochs"]
    dataset = hyper["dataset_type"]
    dataset_size = hyper["dataset_size"]

    log_file_name = generate_log_file_name(hyper)

    if dataset == 1:
        xtr, ytr, xva, yva, xte, yte = load_clothes(dataset_size / 100.0)
    elif dataset_type == 2:
        xtr, ytr, xva, yva, xte, yte = load_faces(dataset_size / 100.0)
    else:
        raise NotImplementedError

    history = model.fit(
        xtr,
        ytr,
        batch_size,
        validation_data = (xva, yva),
        epochs=num_epochs
    )

    # DO NOT eval due to problems when splitting data
    # No samples from all classes when dataset size is to small
    try:
        eval_ = model.evaluate(xte, yte)
        for val, key in zip(eval_, model.metrics_names):
            hyper[key] = val
    except:
        print("Not samples from all classes for evaluation")

    save_log_metrics(log_file_name, hyper, history)
    save_plot_metrics(log_file_name, history)
    model.save_weights(log_file_name + ".hdf")

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
    dataset_size = str(dataset_size).replace(".", "")
    activation = hyper["activation"]
    learning_rate = hyper["learning_rate"]
    learning_rate = str(learning_rate).replace(".", "") #convert to string to avoid decimal point in the filename

    return "./baseline_experiments/{:s}_M_{:d}_A_{:s}_L_{:s}_D_{:s}_S_{:s}".format(exp_name, model_type, activation, learning_rate, dataset_name, dataset_size)

def train_networks(exp_name, model_type, learning_rate, training_size, 
    batch_size, num_epochs, dataset_size, dataset_type, activation):
    model = None

    if model_type == 1:
        model = get_model_1(learning_rate, activation)
    elif model_type == 2:
        model = get_model_2(learning_rate, activation)
    elif model_type == 3:
        model = get_model_3(learning_rate, activation)
    else:
        raise NotImplementedError

    hyper = OrderedDict()
    hyper["learning_rate"] = learning_rate
    hyper["training_size"] = training_size
    hyper["batch_size"] = batch_size
    hyper["num_epochs"] = num_epochs
    hyper["dataset_type"] = dataset_type
    hyper["dataset_size"] = dataset_size
    hyper["model_type"] = model_type
    hyper["exp_name"] = exp_name
    hyper["activation"] = activation

    train_model(model, hyper)

#########################################################################################

parser = argparse.ArgumentParser(description="CNN systems for coursework 3")
parser.add_argument('exp_name', type=str, help="Name of experiment")
parser.add_argument('model_type', type=int, help="Type of classifier")
parser.add_argument('-n', dest='num_epochs', type=int, default=100)
parser.add_argument('-l', dest='learning_rate', type=float, default=0.001)
parser.add_argument('-t', dest='training_size', type=int, default=2000)
parser.add_argument('-b', dest='batch_size', type=int, default=50)
parser.add_argument('-s', dest='dataset_size', type=float, default=100)
parser.add_argument('-d', dest='dataset_type', type=int, default=1)
parser.add_argument('-a', dest='activation', type=str, default="relu")

args = parser.parse_args()
exp_name = args.exp_name
model_type = args.model_type
num_epochs = args.num_epochs
learning_rate = args.learning_rate
training_size = args.training_size
batch_size = args.batch_size
dataset_size = args.dataset_size
dataset_type = args.dataset_type
activation = args.activation

train_networks(exp_name, model_type, learning_rate, training_size, batch_size, num_epochs, dataset_size, dataset_type, activation)