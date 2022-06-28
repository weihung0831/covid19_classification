import matplotlib.pyplot as plt
import numpy as np
import pretty_errors
import tensorflow as tf
from icecream import ic
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import (Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam


def vgg19_model(input_shape):
    model = VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
    inputs = Input(input_shape)
    x = preprocess_input(inputs)
    model.trainable = False
    x = model(x, training=False)
    x = Flatten()(x)
    x = Dropout(rate=0.5)(x)
    outputs = Dense(3, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="CategoricalCrossentropy",
        metrics="accuracy",
    )

    return model


def visualize_training_results(history, history_plot_path):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(212)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.savefig(history_plot_path)


def training(
    input_shape, x_train, y_train, x_val, y_val, model_path, history_plot_path
):
    model = vgg19_model(input_shape)
    history = model.fit(
        x_train, y_train, batch_size=32, epochs=30, validation_data=(x_val, y_val)
    )
    model.save(model_path)
    visualize_training_results(history, history_plot_path)


def main():
    data = np.load(file="./COVID-19_Radiography_Dataset/data.npz")
    x_train, x_val, y_train, y_val = (
        data["train_img"],
        data["val_img"],
        data["train_label"],
        data["val_label"],
    )
    ic(
        x_train.shape,
        x_val.shape,
        y_train.shape,
        y_val.shape,
    )

    training(
        x_train.shape[1:],
        x_train,
        y_train,
        x_val,
        y_val,
        model_path="./model/model.tf",
        history_plot_path="./model/history.png",
    )


if __name__ == "__main__":
    main()
