import numpy as np
from PIL import Image
import scipy
import matplotlib.pyplot as plt

def standardize(img):
    #padding
    longer_side = max(img.size)
    horizontal_padding = (longer_side - img.size[0]) / 2
    vertical_padding = (longer_side - img.size[1]) / 2
    img = img.crop(
        (
            -horizontal_padding,
            -vertical_padding,
            img.size[0] + horizontal_padding,
            img.size[1] + vertical_padding
        )
    )
    # resizing to standardized size
    img = img.resize([64,64],Image.ANTIALIAS) # Might want to change it up to 128 in future
    # plt.imshow(img)
    
    # converting image to numpy array
    img.load()
    img = np.asarray(img, dtype="int32")
    return img

def load_train_id_cards():
    X_train = np.empty((482,64,64,3), dtype="int32")
    Y_train = np.empty(shape=(482,4),dtype="int32")
    for i in range(1,483):
        img = Image.open("id_cards/"+str(i)+".jpg")
        img = standardize(img)
        #print(i, img.shape)
        X_train[i-1] = img
        Y_train[i-1] = np.array([1,0,0,0])
    return X_train,Y_train

def load_train_slides():
    X_train = np.empty((316,64,64,3), dtype="int32")
    Y_train = np.empty(shape=(316,4),dtype="int32")
    for i in range(1,317):
        img = Image.open("slides/"+str(i)+".jpg")
        img = standardize(img)
        #print(i,img.shape)
        X_train[i-1] = img
        Y_train[i-1] = np.array([0,1,0,0])
    return X_train,Y_train

def load_train_paper_documents():
    X_train = np.empty((306,64,64,3), dtype="int32")
    Y_train = np.empty(shape=(306,4),dtype="int32")
    for i in range(1,307):
        img = Image.open("paper_documents/"+str(i)+".jpg")
        img = standardize(img)
        #print(i, img.shape)
        X_train[i-1] = img
        Y_train[i-1] = np.array([0,0,1,0])
    return X_train,Y_train

def load_train_receipts():
    X_train = np.empty((300,64,64,3), dtype="int32")
    Y_train = np.empty(shape=(300,4),dtype="int32")
    for i in range(1,301):
        img = Image.open("receipts/"+str(i)+".jpg")
        img = standardize(img)
        #print(i, img.shape)
        X_train[i-1] = img
        Y_train[i-1] = np.array([0,0,0,1])
    return X_train,Y_train

def load_test_id_cards():
    X_test = np.empty((24,64,64,3), dtype="int32")
    Y_test = np.empty(shape=(24,4),dtype="int32")
    for i in range(1,25):
        img = Image.open("id_cards_test/"+str(i)+".jpg")
        img = standardize(img)
        X_test[i-1] = img
        Y_test[i-1] = np.array([1,0,0,0])
    return X_test,Y_test

def load_test_slides():
    X_test = np.empty((10,64,64,3), dtype="int32")
    Y_test = np.empty(shape=(10,4),dtype="int32")
    for i in range(1,11):
        img = Image.open("slides_test/"+str(i)+".jpg")
        img = standardize(img)
        X_test[i-1] = img
        Y_test[i-1] = np.array([0,1,0,0])
    return X_test,Y_test

def load_test_paper_documents():
    X_test = np.empty((14,64,64,3), dtype="int32")
    Y_test = np.empty(shape=(14,4),dtype="int32")
    for i in range(1,15):
        img = Image.open("paper_documents_test/"+str(i)+".jpg")
        img = standardize(img)
        X_test[i-1] = img
        Y_test[i-1] = np.array([0,0,1,0])
    return X_test,Y_test

def load_test_receipts():
    X_test = np.empty((17,64,64,3), dtype="int32")
    Y_test = np.empty(shape=(17,4),dtype="int32")
    for i in range(1,18):
        img = Image.open("receipts_test/"+str(i)+".jpg")
        img = standardize(img)
        X_test[i-1] = img
        Y_test[i-1] = np.array([0,0,0,1])
    return X_test,Y_test