import pandas as pd
from icecream import ic
import os
import shutil
import cv2
import tqdm


def load_data(data):
    train_df = pd.read_csv(data, sep=' ', header=None)
    train_df.columns = ['patient id', 'filename', 'class', 'source']
    train_df.drop(['patient id', 'source'], axis=1, inplace=True)
    
    return train_df


def labeling(data, outpit_negative_dir, output_positive_dir):
    negative = data[data['class'] == 'negative']
    positive = data[data['class'] == 'positive']
    
    # negative.to_csv(outpit_negative_dir, index=False)
    # positive.to_csv(output_positive_dir, index=False)
    
    return negative, positive


if __name__ == '__main__':
    train_df = './data/train.txt'
    test_df = './data/test.txt'

    train_df = load_data(train_df)
    tesst_df = load_data(test_df)
    ic(train_df)
    ic(tesst_df)
    ic(train_df['class'].value_counts())
    
    train_label = labeling(train_df, './data/train_negative.csv', './data/train_positive.csv')
    test_label = labeling(tesst_df, './data/test_negative.csv', './data/test_positive.csv')
    ic(train_label)
    ic(test_label)
