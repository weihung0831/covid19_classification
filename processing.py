import os

import cv2
import numpy as np
import pretty_errors
from icecream import ic
from sklearn.preprocessing import OneHotEncoder


def load_img(dir, label_num):
    image, label = [], []
    image_files = os.listdir(dir)
    ic(len(image_files))
    image = np.zeros((len(image_files), 180, 180, 1))
    label = np.ones((len(image_files), 1)) * label_num
    
    i = 0
    for files in image_files:
        ic(files)
        path = os.path.join(dir, files)
        img = cv2.imread(path, 0)
        # blur = cv2.bilateralFilter(img, 9, 75, 75)
        # dst = cv2.equalizeHist(blur)
        resize = cv2.resize(img, (180, 180))
        resize = np.expand_dims(resize, axis=-1)
        ic(resize.shape)
        image[i] = resize
        i += 1

    return image, label

        
def concatenate_image(train_negative_img, train_positive_img):
    train_x = np.concatenate([train_negative_img, train_positive_img])
    # ic(train_x)
    return train_x


def concatenate_label(train_negative_label, train_positive_label):
    train_y = np.concatenate([train_negative_label, train_positive_label])
    # ic(train_y)
    return train_y


def load_train_data(train_negative_dir, train_positive_dir):
    train_negative_img, train_negative_label = load_img(train_negative_dir, 0)
    train_positive_img, train_positive_label = load_img(train_positive_dir, 1)
    ic(len(os.listdir(train_negative_dir)))
    ic(len(os.listdir(train_positive_dir)))
    
    train_x = concatenate_image(train_negative_img, train_positive_img)
    train_y = concatenate_label(train_negative_label, train_positive_label)
    # ic(train_x)
    # ic(train_y)
    
    np.random.seed(123)
    np.random.shuffle(train_x)
    np.random.seed(123)
    np.random.shuffle(train_y)
    # ic(train_x)
    # ic(train_y)
    
    return train_x, train_y


def load_test_data(test_negative_dir, test_positive_dir):
    test_negative_img, test_negative_label = load_img(test_negative_dir, 0)
    test_positive_img, test_positive_label = load_img(test_positive_dir, 1)
    # ic(len(os.listdir(test_negative_dir)))
    # ic(len(os.listdir(test_positive_dir)))
    
    file_list = []
    for dir in [test_negative_dir, test_positive_dir]:
        file_list += sorted(os.listdir(dir))
    
    test_img = np.concatenate([test_negative_img, test_positive_img])
    y_true = np.concatenate([test_negative_label, test_positive_label])
    # ic(test_img)
    # ic(y_true)
    
    return test_img, y_true, file_list
