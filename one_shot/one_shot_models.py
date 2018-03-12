import argparse

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2

def load_dataset(file_train, file_test, size=1.0):
    # loadtxt gives results with that b' char
    # data_train = np.loadtxt(file_train, dtype='str', delimiter=',')
    # data_test = np.loadtxt(file_test, dtype='str', delimiter=',')

    data_train = np.genfromtxt(file_train, dtype='str', delimiter=',')
    data_test = np.genfromtxt(file_test, dtype='str', delimiter=',')

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

    return xtr, ytr, xva, yva, xte, yte

def get_model_1(learning_rate):
    VGG=VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(64,64,1), pooling=None)
    model = Sequential()
    for l in VGG.layers:
        model.add(l)
    model.add(Flatten(input_shape=VGG.output_shape[1:]))
    model.add(Dense(7, activation='softmax'))

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[19:20]:
        layer.trainable = True

    optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False )
    model.compile(optimizer='SGD', loss="categorical_crossentropy", metrics=["accuracy"])


    return model


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

def get_images(paths):

    images = []

    for path in paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)

    return np.array(images)

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

    xtr = get_images(xtr)
    xte = get_images(xte)
    xva = get_images(xva)

    history = model.fit(
        xtr,
        ytr,
        batch_size,
        validation_data = (xva, yva),
        epochs=num_epochs
    )

    eval_ = model.evaluate(xte, yte)
    for val, key in zip(eval_, model.metrics_names):
        hyper[key] = val

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
    return "{:s}_model_{:d}_{:s}_{:d}".format(exp_name, model_type, dataset_name, dataset_size)

def train_networks(exp_name, model_type, learning_rate, training_size, batch_size, num_epochs, dataset_size, dataset_type):
    model = None

    if model_type == 1:
        model = get_model_1(learning_rate)
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

    train_model(model, hyper)

#########################################################################################

parser = argparse.ArgumentParser(description="CNN systems for coursework 3")
parser.add_argument('exp_name', type=str, help="Name of experiment")
parser.add_argument('model_type', type=int, help="Type of classifier")
parser.add_argument('-n', dest='num_epochs', type=int, default=100)
parser.add_argument('-l', dest='learning_rate', type=float, default=0.001)
parser.add_argument('-t', dest='training_size', type=int, default=2000)
parser.add_argument('-b', dest='batch_size', type=int, default=50)
parser.add_argument('-s', dest='dataset_size', type=int, default=100)
parser.add_argument('-d', dest='dataset_type', type=int, default=1)

args = parser.parse_args()
exp_name = args.exp_name
model_type = args.model_type
num_epochs = args.num_epochs
learning_rate = args.learning_rate
training_size = args.training_size
batch_size = args.batch_size
dataset_size = args.dataset_size
dataset_type = args.dataset_type

train_networks(exp_name, model_type, learning_rate, training_size, batch_size, num_epochs, dataset_size, dataset_type)
