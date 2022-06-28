import matplotlib.pyplot as plt
import pretty_errors
import tensorflow as tf
from icecream import ic
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import load_model
import numpy as np


def test_model(test_img, y_true, class_name, model_path, confusion_matrix_plot_path):
    model = load_model(model_path)
    y_pred = model.predict(test_img)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    ic(accuracy_score(y_true, y_pred))
    ic(classification_report(y_true, y_pred, target_names=class_name))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred), display_labels=class_name
    ).plot()
    plt.savefig(confusion_matrix_plot_path)


def main():
    data = np.load(file="./COVID-19_Radiography_Dataset/data.npz")
    x_test, y_test = (
        data["test_img"],
        data["test_label"],
    )
    ic(x_test.shape, y_test.shape)

    class_name = ["covid", "normal", "pneumonia"]
    test_model(
        x_test,
        y_test,
        class_name,
        model_path="./model/model.tf/",
        confusion_matrix_plot_path="./model/confusion_matrix.png",
    )


if __name__ == "__main__":
    main()
