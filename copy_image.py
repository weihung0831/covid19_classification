import pandas as pd
from icecream import ic
import os
import shutil


def load_data(data):
    train_df = pd.read_csv(data, sep=' ', header=None)
    train_df.columns = ['patient id', 'filename', 'class', 'source']
    train_df.drop(['patient id', 'source'], axis=1, inplace=True)
    # train_df.to_csv('./data/train.csv', index=False)
    
    return train_df


def make_dir(class_name, output_dir):
    for label_name in class_name:
        ic(label_name)
        os.makedirs(os.path.join(output_dir, label_name))
        print('Successfully created directory')
        
    return class_name


def copy_image(class_name, class_label, input_dir, output_dir):
    for label_name in class_name:
        ic(label_name)
        for img_name in list(class_label[class_label['class'] == label_name]['filename']):
            ic(img_name)
            img = os.path.join(input_dir, img_name)
            copy_img = shutil.copy(img, output_dir)


if __name__ == '__main__':
    train_negative = './data/train_negative.csv'
    train_positive = './data/train_positive.csv'
    test_negative = './data/test_negative.csv'
    test_positive = './data/test_positive.csv'
    train_dir = './data/train/'
    test_dir = './data/test/'

    train_negative_labels = pd.read_csv(train_negative)
    train_positive_labels = pd.read_csv(train_positive)
    test_negative_labels = pd.read_csv(test_negative)
    test_positive_labels = pd.read_csv(test_positive)
    # ic(train_negative_labels)
    # ic(train_positive_labels)
    # ic(test_negative_labels)
    # ic(test_positive_labels)
    
    train_negative_class_name = list(train_negative_labels['class'].unique())
    train_positive_class_name = list(train_positive_labels['class'].unique())
    test_negative_class_name = list(test_negative_labels['class'].unique())
    test_positive_class_name = list(test_positive_labels['class'].unique())
    # ic(train_negative_class_name)
    # ic(train_positive_class_name)
    # ic(test_negative_class_name)
    # ic(test_positive_class_name)
    
    make_train_negative_dir = make_dir(train_negative_class_name, './dataset/train/')
    make_train_positive_dir = make_dir(train_positive_class_name, './dataset/train/')
    make_test_negative_dir = make_dir(test_negative_class_name, './dataset/test/')
    make_test_positive_dir = make_dir(test_positive_class_name, './dataset/test/')
    
    copy_train_negative_img = copy_image(train_negative_class_name, train_negative_labels, train_dir, './dataset/train/negative')
    copy_train_positive_img = copy_image(train_positive_class_name, train_positive_labels, train_dir, './dataset/train/positive')
    copy_test_negative_img = copy_image(test_negative_class_name, test_negative_labels, test_dir, './dataset/test/negative')
    copy_test_positive_img = copy_image(test_positive_class_name, test_positive_labels, test_dir, './dataset/test/positive')            
    
    ic(len(os.listdir('./dataset/train/negative')))
    ic(len(os.listdir('./dataset/train/positive')))
    ic(len(os.listdir('./dataset/test/negative')))
    ic(len(os.listdir('./dataset/test/positive')))
    