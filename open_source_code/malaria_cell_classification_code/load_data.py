############Load libraries#####################################################
import cv2
import numpy as np
import os
from keras.utils import np_utils
###############################################################################
#cross-validation at the patient level
train_data_dir = 'f1_mal/train'
valid_data_dir = 'f1_mal/valid'
###############################################################################
# declare the number of samples in each category
nb_train_samples = 22045 #  training samples
nb_valid_samples = 5513#  validation samples
num_classes = 2
img_rows_orig = 100
img_cols_orig = 100

def load_training_data():
    # Load training images
    labels = os.listdir(train_data_dir)
    total = len(labels)
    
    X_train = np.ndarray((nb_train_samples, img_rows_orig, img_cols_orig, 3), dtype=np.uint8)
    Y_train = np.zeros((nb_train_samples,), dtype='uint8')
    

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    j = 0
    for label in labels:
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        total = len(image_names_train)
        print(label, total)
        for image_name in image_names_train:
            img = cv2.imread(os.path.join(train_data_dir, label, image_name), cv2.IMREAD_COLOR)
            img = np.array([img])
            X_train[i] = img
            Y_train[i] = j

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        j += 1    
    print(i)                
    print('Loading done.')
    
    print('Transform targets to keras compatible format.')
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)

    np.save('imgs_train.npy', X_train, Y_train)
    return X_train, Y_train
    
def load_validation_data():
    # Load validation images
    labels = os.listdir(valid_data_dir)
    

    X_valid = np.ndarray((nb_valid_samples, img_rows_orig, img_cols_orig, 3), dtype=np.uint8)
    Y_valid = np.zeros((nb_valid_samples,), dtype='uint8')

    i = 0
    print('-'*30)
    print('Creating validation images...')
    print('-'*30)
    j = 0
    for label in labels:
        image_names_valid = os.listdir(os.path.join(valid_data_dir, label))
        total = len(image_names_valid)
        print(label, total)
        for image_name in image_names_valid:
            img = cv2.imread(os.path.join(valid_data_dir, label, image_name), cv2.IMREAD_COLOR)

            img = np.array([img])

            X_valid[i] = img
            Y_valid[i] = j

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        j += 1
    print(i)            
    print('Loading done.')
    
    print('Transform targets to keras compatible format.');
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    np.save('imgs_valid.npy', X_valid, Y_valid)
    
    return X_valid, Y_valid

def load_resized_training_data(img_rows, img_cols):

    X_train, Y_train = load_training_data()
    # Resize trainging images
    X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
    
    return X_train, Y_train
    
def load_resized_validation_data(img_rows, img_cols):

    X_valid, Y_valid = load_validation_data()
       
    # Resize images
    X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])
        
    return X_valid, Y_valid   