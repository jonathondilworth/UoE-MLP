from random import randrange
import numpy as np
import cv2

def load_clothes(size=1.0):
    return load_dataset("./clothes_test.txt", size)
    
def load_faces(size=1.0):
    return load_dataset("./faces_test.txt", size)

def load_dataset(file_val, size=1.0):
    data_val = np.genfromtxt(file_val, dtype='str', delimiter=',')
    xva, yva = data_val[:,1], data_val[:,0].astype(int)
    
    return xva, yva

def open_image(path, augment=True, soft=True):
    image_path = path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if augment == False or randrange(20) == 0:
        return image.reshape((64, 64, 1))

    # Rotation
    if randrange(3) == 0:
        if soft:
            angle = randrange(15, 35)
        else:
            angle = randrange(30, 50)
        image = rotate(image, angle)

    # Blur
    if randrange(4) == 0:
        if soft:
            kernel = (5,5)
        else:
            kernel = (7,7)
        image = cv2.blur(image, kernel)

    # Skew
    if randrange(2) == 0:
        side = randrange(4)
        rev = randrange(2)
        if soft:
            percentage = randrange(10, 30)
        else:
            percentage = randrange(25, 45)

        image = skew(image, side, percentage, rev == 0)

    # Shift
    if randrange(3) == 0:
        if soft:
            percentage = randrange(0, 15)
        else:
            percentage = randrange(10, 20)

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
        pts1 = np.float32([[0,sd],[w-1,0],[0,h-1-sd],[w-1,h-1]])
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

def augment():
    xf, yf = load_faces()

    for path in xf:
        image = open_image(path, True, False)
        cv2.imwrite(path, image)

    xc, yc = load_clothes()

    for path in xc:
        image = open_image(path, True, False)
        cv2.imwrite(path, image)

augment()