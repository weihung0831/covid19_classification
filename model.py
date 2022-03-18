import pretty_errors
import tensorflow as tf


# https://link.springer.com/article/10.1007%2Fs42600-020-00120-5
def model1():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    inputs = tf.keras.layers.Input(shape=(180, 180, 1))
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=5, activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.Dense(units=32, activation='relu')(x)
    # x = tf.keras.layers.Dropout(rate=0.2)(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='model.png')

    model.summary()

    return model


def model2():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    inputs = tf.keras.layers.Input(shape=(180, 180, 1))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=32, activation='relu')(x)
    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


# https://link.springer.com/content/pdf/10.1007/s42979-021-00881-5.pdf
def model3():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    inputs = tf.keras.layers.Input(shape=(180, 180, 3))
    # x = inputs / 255.
    x = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', pooling='max')(inputs)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=4096, activation='relu')(x)
    x = tf.keras.layers.Dense(units=4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


# model = tf.keras.models.load_model('./model/model1/model_13_0.962.tf/')
model = model1()
# model.summary()

# tf.keras.utils.plot_model(model, show_shapes=True, to_file='model.png')
