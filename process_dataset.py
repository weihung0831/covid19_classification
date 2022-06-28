import os

import cv2
import numpy as np
import pandas as pd
import pretty_errors
from icecream import ic
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


def load_data(data, class_num, class_name):
    filename = data["FILE NAME"] + ".png"

    label = []
    for i in range(len(filename)):
        label_num = np.array([class_num])
        label.append(label_num)
    label = np.array(label).flatten()
    data = pd.DataFrame({"filename": filename, "label": label})
    label = []
    for i in data["label"]:
        if i == class_num:
            i = class_name
        label.append(i)
    data["label"] = label

    return data


def load_image(data, directory):
    image = []
    for i in tqdm(data["filename"]):
        path = os.path.join(directory, i)
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (256, 256))
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.equalizeHist(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        image.append(img)
    image = np.array(image, dtype="float32")

    return image


def load_label(data, class_name, class_num):
    label = []
    for i in data["label"]:
        if i == class_name:
            i = class_num
        label.append(i)
    data["label"] = label
    label = np.array(label, dtype="float32")

    return label


def main():
    covid_data = pd.read_excel("./COVID-19_Radiography_Dataset/COVID.metadata.xlsx")
    normal_data = pd.read_excel("./COVID-19_Radiography_Dataset/Normal.metadata.xlsx")
    pneumonia_data = pd.read_excel(
        "./COVID-19_Radiography_Dataset/Viral Pneumonia.metadata.xlsx"
    )

    covid_data = load_data(covid_data, class_num=0, class_name="covid")
    normal_data = load_data(normal_data, class_num=1, class_name="normal")
    pneumonia_data = load_data(pneumonia_data, class_num=2, class_name="pneumonia")

    covid_data = covid_data[:1184]
    normal_data = normal_data[:1319]
    pneumonia_data = pneumonia_data[:1294]

    train_covid_data = covid_data[:852]
    train_normal_data = normal_data[:949]
    train_pneumonia_data = pneumonia_data[:932]

    val_covid_data = covid_data[852:946]
    val_normal_data = normal_data[949:1051]
    val_pneumonia_data = pneumonia_data[932:1040]

    test_covid_data = covid_data[946:]
    test_normal_data = normal_data[1051:]
    test_pneumonia_data = pneumonia_data[1040:]

    covid_img_directory = "./COVID-19_Radiography_Dataset/COVID/"
    normal_img_directory = "./COVID-19_Radiography_Dataset/Normal/"
    pneumonia_img_directory = "./COVID-19_Radiography_Dataset/Viral Pneumonia/"

    train_covid_img = load_image(train_covid_data, covid_img_directory)
    train_normal_img = load_image(train_normal_data, normal_img_directory)
    train_pneumonia_img = load_image(train_pneumonia_data, pneumonia_img_directory)

    val_covid_img = load_image(val_covid_data, covid_img_directory)
    val_normal_img = load_image(val_normal_data, normal_img_directory)
    val_pneumonia_img = load_image(val_pneumonia_data, pneumonia_img_directory)

    test_covid_img = load_image(test_covid_data, covid_img_directory)
    test_normal_img = load_image(test_normal_data, normal_img_directory)
    test_pneumonia_img = load_image(test_pneumonia_data, pneumonia_img_directory)

    train_covid_label = load_label(train_covid_data, class_name="covid", class_num=0)
    train_normal_label = load_label(train_normal_data, class_name="normal", class_num=1)
    train_pneumonia_label = load_label(
        train_pneumonia_data, class_name="pneumonia", class_num=2
    )

    val_covid_label = load_label(val_covid_data, class_name="covid", class_num=0)
    val_normal_label = load_label(val_normal_data, class_name="normal", class_num=1)
    val_pneumonia_label = load_label(
        val_pneumonia_data, class_name="pneumonia", class_num=2
    )

    test_covid_label = load_label(test_covid_data, class_name="covid", class_num=0)
    test_normal_label = load_label(test_normal_data, class_name="normal", class_num=1)
    test_pneumonia_label = load_label(
        test_pneumonia_data, class_name="pneumonia", class_num=2
    )

    x_train = np.concatenate([train_covid_img, train_normal_img, train_pneumonia_img])
    x_val = np.concatenate([val_covid_img, val_normal_img, val_pneumonia_img])
    x_test = np.concatenate([test_covid_img, test_normal_img, test_pneumonia_img])

    y_train = np.concatenate(
        [train_covid_label, train_normal_label, train_pneumonia_label]
    )
    y_val = np.concatenate([val_covid_label, val_normal_label, val_pneumonia_label])
    y_test = np.concatenate([test_covid_label, test_normal_label, test_pneumonia_label])

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_train = OneHotEncoder().fit(y_train).transform(y_train).toarray()
    y_val = OneHotEncoder().fit(y_val).transform(y_val).toarray()
    y_test = OneHotEncoder().fit(y_test).transform(y_test).toarray()

    np.savez(
        "./COVID-19_Radiography_Dataset/data.npz",
        train_img=x_train,
        val_img=x_val,
        test_img=x_test,
        train_label=y_train,
        val_label=y_val,
        test_label=y_test,
    )


if __name__ == "__main__":
    main()
