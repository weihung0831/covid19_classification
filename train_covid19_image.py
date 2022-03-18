import time
import pretty_errors
import tensorflow as tf
from icecream import ic
from plot import training_model_plot
from model import model1, model2, model3
from processing import load_train_data
from train_model_and_predict import compile_and_train_model

time_start = time.time()

train_x, train_y = load_train_data(train_negative_dir='dataset/train/negative/',
                                   train_positive_dir='dataset/train/positive/')
model = model1()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='BinaryCrossentropy', metrics=['accuracy'])
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=0, mode='max'),
             tf.keras.callbacks.ModelCheckpoint(filepath='./model/model3/model_{epoch:02d}_{val_loss:.2f}.tf',
                                                save_best_only=True)]
history = model.fit(x=train_x, y=train_y, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks)
training_model_plot(history)
# model = compile_and_train_model(model3(), train_x / 255., train_y)

time_end = time.time()
spend_time = ((time_end - time_start).__format__('.2f')) + 's'
ic(spend_time)
