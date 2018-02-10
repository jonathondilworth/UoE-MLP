import numpy as np
import cv2

def load_clothes():
    train = np.loadtxt("./clothes_raw/fashion-mnist_train.csv", skiprows=1, dtype=int, delimiter=',')
    test = np.loadtxt("./clothes_raw/fashion-mnist_test.csv", skiprows=1, dtype=int, delimiter=',')
    
    # testing of code in notebook
    #train = train[:100, :]
    #test = test[:100, :]
    
    return train, test

def split_xy(data):
    return data[:,1:], data[:,0]

def resize_clothes(x):
    new_set = np.zeros(4096)
    
    for sample in x:
        sample = sample.astype(float)
        resized = cv2.resize(sample.reshape(28, 28), (64, 64))
        new_set = np.vstack((new_set, resized.ravel()))
        
    return new_set[1:]

def save(x, y, name):
    file = open(name, "wb")
    np.savez(file, inputs=x, targets=y)

def filter(data, invalid_labels):
    
    for label in invalid_labels:
        data = data[data[:,0]!=label, :]

    return data

def relabel(y):
    current_labels = np.unique(y)
    new_y = np.ones_like(y)*(-1)

    for idx, label in enumerate(current_labels):
        new_y[y == label] = idx

    return new_y.astype(int)

def save_images(fname, x, y):

    records = np.array(["-1", "fakepath"]).astype(str)

    c = 0
    for label, image in zip(y, x):
        image_path = "./clothes/{:s}/{:d}/{:d}.jpg".format(fname,label,c)
        cv2.imwrite(image_path, image.reshape((28, 28)))
        records = np.vstack((records, [label, image_path]))
        c = c + 1

    records = records[1:,:]
    np.savetxt("clothes_{:s}.txt".format(fname), records, delimiter=',', fmt='%s')

def prepare():
    train, test = load_clothes()

    # remove samples of classes 7, 6, 4
    invalid_labels = [4,6,7]

    train = filter(train, invalid_labels)
    test = filter(test, invalid_labels)

    xtr_, ytr_ = split_xy(train)
    xte_, yte_ = split_xy(test)

    # to have a continuos set of labels: 0, 1, 2, 3,...
    ytr_ = relabel(ytr_)
    yte_ = relabel(yte_)
    assert -1 not in np.unique(ytr_) 
    assert -1 not in np.unique(yte_) 

    print("before save images")
    save_images("train", xtr_, ytr_)
    save_images("test", xte_, yte_)
    

prepare()