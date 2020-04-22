import numpy as np
import pandas as pd
import os
import sys
import datetime
import requests
import json
import importlib
api = importlib.import_module("API_Puller")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

### Debug logging
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

_batch_size = 1
_buffer_size = 10000

_max_epochs = 400
_back_in_time = 60 # Days
_step = 6 # Days to offset next dataset
_target_size = 1 # How many to predict



### Optimizers
_optimizer = keras.optimizers.Nadam()
# keras.optimizers.RMSprop()
# keras.optimizers.Nadam()
# keras.optimizers.Adam()

### Losses
_loss = keras.losses.mean_absolute_error
# keras.losses.mean_squared_error
# keras.losses.mean_absolute_error

train = api.pulldata()



### Handle Data ###
def handle_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    
    data = []
    labels = []

    for i in range(0, len(dataset) - history_size, step):
        seq = dataset[i:i + history_size]
        label = target[i + history_size - 1]
        data.append(seq)
        labels.append(label)

    return data, labels

scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(train[:,:-1])

print("Making timestep sets (Step size: %s, History: %s days, Target value size: %s day(s))" % (_step, _back_in_time, _target_size))

X, y = handle_data(
    rescaledX, train[:, 1], 
    0, 
    len(train), 
    _back_in_time, 
    _target_size,
    _step, 
    single_step=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)
print("Made", len(y), "datasets...")

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.cache().shuffle(_buffer_size).batch(_batch_size).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.cache().shuffle(_buffer_size).batch(_batch_size).repeat()



### Models
model_std = Sequential([
    LSTM(50, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(50),
    Dense(1)
])

model_wide = Sequential([
    LSTM(300, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(300),
    Dense(1)
])

model_mega_wide = Sequential([
    LSTM(600, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(600),
    Dense(1)
])

model_deep = Sequential([
    LSTM(50, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(50, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(50, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(50, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(50),
    Dense(1)
])

model_shallow_deep = Sequential([
    LSTM(50, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(50, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(50),
    Dense(1)
])

model_wide_deep = Sequential([
    LSTM(300, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(300, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(300, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(300, input_shape=X_train[0].shape, return_sequences=True),
    LSTM(300),
    Dense(1)
])



### Define callbacks
def get_callbacks(name):
    return [
        EarlyStopping(monitor="val_loss", patience=10),
        TensorBoard("logs/" + name)
    ]



### Compile and Fit
def compile_and_fit(model, name=None, optimizer=_optimizer, loss=_loss, max_epochs=_max_epochs):
    model.compile(loss=loss, optimizer=optimizer)

    model.summary()

    print("\nTraining model...")
    model_history = model.fit(
        x=train_dataset, 
        epochs=max_epochs, 
        steps_per_epoch=len(y_train), 
        validation_data=val_dataset, 
        validation_steps=len(y_val), 
        verbose=1, 
        callbacks=get_callbacks(name))
    
    return model_history

fitted_models = {}

print("\n")

print("Beginning trainning of Model:", "std")
fitted_models["std"] = compile_and_fit(model_std, "std")

print("Beginning trainning of Model:", "wide")
fitted_models["wide"] = compile_and_fit(model_wide, "wide")

print("Beginning trainning of Model:", "mega_wide")
fitted_models["mega_wide"] = compile_and_fit(model_mega_wide, "mega_wide")

print("Beginning trainning of Model:", "deep")
fitted_models["deep"] = compile_and_fit(model_deep, "deep")

print("Beginning trainning of Model:", "model_shallow_deep")
fitted_models["model_shallow_deep"] = compile_and_fit(model_shallow_deep, "model_shallow_deep")

print("Beginning trainning of Model:", "wide_deep")
fitted_models["wide_deep"] = compile_and_fit(model_wide_deep, "wide_deep")

# Links
# https://www.tensorflow.org/api_docs/python/tf/keras/Model?version=nightly#fit
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard?version=nightly
# https://www.tensorflow.org/install
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping?version=nightly
# https://tensorboard.dev/#get-started
# https://github.com/pytorch/pytorch/issues/22676
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#training_procedure
# https://github.com/Createdd/Writing/blob/master/2018/articles/DebugTFBasics.md#2-use-the-tfprint-operation
# https://github.com/haribaskar/Keras_Cheat_Sheet_Python
# https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage
# https://www.tensorflow.org/api_docs/python/tf/executing_eagerly
# https://docs.docker.com/storage/volumes/
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/MeanSquaredError?version=nightly
# https://www.tensorflow.org/api_docs/python/tf/keras/Model?version=nightly#fit
# https://stackoverflow.com/questions/31448821/how-to-write-data-to-host-file-system-from-docker-container
# https://github.com/tensorflow/tensorflow/issues/7652
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
# https://stackoverflow.com/questions/509211/understanding-slice-notation
# https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/