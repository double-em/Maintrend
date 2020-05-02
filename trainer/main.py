import numpy as np
import pandas as pd
import os
import sys
import datetime
import requests
import json
import importlib
from pathlib import Path
api = importlib.import_module("API_Puller")
util = importlib.import_module("util")

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
from tensorflow.python import debug as tf_debug
from tensorboard.plugins.hparams import api as hp


model_name = "test"
model_version = 3

time_now_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs/"
models_dir = "models/"

print("\nVisible Devices:", tf.config.get_visible_devices())

_patience = 60

_batch_size = 16
_buffer_size = 10000

_max_epochs = 1000
_back_in_time = 60 # Days
_step = 1 # Days to offset next dataset
_target_size = 1 # How many to predict



### Hyperparamters
hp_hidden_num_layers = hp.HParam('hidden_num_layers', hp.IntInterval(0, 4))
hp_optimizer = hp.HParam('optimizer', hp.Discrete(['nadam', 'adam', 'rmsprop', 'sgd']))
hp_output_units = hp.HParam('output_units', hp.Discrete([50, 300, 600]))

hypertune = False

hp.hparams_config(
    hparams=[hp_hidden_num_layers, hp_optimizer, hp_output_units],
    metrics=[hp.Metric('mae', display_name="Mean Absolute Error")]
)



### Build model
build_mode = True

if build_mode:
    hparams = {
        hp_hidden_num_layers: 3,
        hp_optimizer: 'rmsprop',
        hp_output_units: 300
    }

### Optimizers
_optimizer = keras.optimizers.Nadam()
# keras.optimizers.RMSprop()
# keras.optimizers.Nadam()
# keras.optimizers.Adam()

### Losses
_loss = keras.losses.mean_absolute_error
# _loss = keras.losses.mean_squared_error

# train_old = api.pulldata2()
train = api.apicallv3(_back_in_time)
trian_length = len(list(train.as_numpy_iterator()))
train_size = int(0.8 * trian_length)
val_size = int(0.1 * trian_length)
test_size = int(0.1 * trian_length)

full_dataset = train.shuffle(trian_length, seed=42)
test_dataset = full_dataset.skip(train_size)

train_dataset = full_dataset.take(train_size).batch(_batch_size, drop_remainder=True).cache().prefetch(train_size)
val_dataset = test_dataset.take(val_size).batch(_batch_size, drop_remainder=True).cache().prefetch(val_size)
test_dataset = test_dataset.skip(val_size).batch(_batch_size, drop_remainder=True).cache().prefetch(test_size)



### Define callbacks
def get_callbacks(name, hparams):
    log_dir_path = log_dir + str(model_version) + "/" + name
    return [
        EarlyStopping(monitor="val_loss", patience=_patience, restore_best_weights=True),
        TensorBoard(
            log_dir=log_dir_path,
            histogram_freq=1
        ),
        hp.KerasCallback(log_dir_path, hparams, name)
    ]



### Compile and Fit
def compile_and_fit(model, name, hparams, optimizer=_optimizer, loss=_loss, max_epochs=_max_epochs):
    model.compile(loss=loss, optimizer=optimizer)

    #model.summary()
    print("Optimizer:", model.optimizer)

    print("\nTraining model...")

    model_history = model.fit(
        train_dataset, 
        epochs=max_epochs, 
        #steps_per_epoch=len(y_train), 
        validation_data=val_dataset, 
        #validation_steps=len(y_val),
        #validation_split=0.10,
        verbose=1, 
        callbacks=get_callbacks(name, hparams))
    
    return model_history

print("\n\n")



### Dynamic model builder
def model_builder(name, hparams):

    model = Sequential(name=name)

    if hparams[hp_hidden_num_layers] == 0:
        model.add(LSTM(hparams[hp_output_units]))
    else:
        model.add(LSTM(hparams[hp_output_units], return_sequences=True))

    for i in range(hparams[hp_hidden_num_layers]):
        if i == (hparams[hp_hidden_num_layers] - 1):
            model.add(LSTM(hparams[hp_output_units]))
        else:
            model.add(LSTM(hparams[hp_output_units], return_sequences=True))

    model.add(Dense(1))

    return model



### Trainer loop
if hypertune:
    session_version = 0

    for output_units in hp_output_units.domain.values:
        for hidden_num_layers in range(hp_hidden_num_layers.domain.min_value, (hp_hidden_num_layers.domain.max_value + 1)):
            for optimizer in hp_optimizer.domain.values:
                hparams = {
                    hp_hidden_num_layers: hidden_num_layers,
                    hp_optimizer: optimizer,
                    hp_output_units: output_units
                }

                print("Starting session:", session_version)
                print({h.name: hparams[h] for h in hparams})

                model_tmp = model_builder(str(session_version), hparams)
                compile_and_fit(model_tmp, model_tmp.name, hparams, hparams[hp_optimizer])

                session_version += 1

if build_mode:
    model_temp = model_builder("prod", hparams)
    compile_and_fit(model_temp, model_temp.name, hparams, hparams[hp_optimizer])

    print("\nSaving model:", model_temp.name)
    # Model version is NEEDED or Tensorflow Serve cant find any "serverable versions"
    save_path = "%s/%s/%s" % (models_dir, model_temp.name, str(model_version))
    model_temp.save(save_path)

    ### Test model
    treshold = 3
    differ = util.DifferenceHolder(treshold)

    predictions = model_temp.predict(test_dataset)

    # Shape = (6, 2, 16, 60, 4)
    # Shape = (batches, (x and y), batch_size, history_size, parameters)
    dataset_list = list(test_dataset.as_numpy_iterator())

    x_s = []
    y_s = []

    for item in dataset_list:
        for ite in item[0]:
            x_s.append(ite)

        for ite in item[1]:
            y_s.append(ite)

    for i in range(len(predictions)):

        prediction = predictions[i][0]
        dataset = x_s[i]
        true_value = y_s[i]

        differ.difference_calc(prediction, true_value, dataset)

        i += 1
    
    util.PrintFinal(differ)



# NOTE: No space on docker volumes = "Fail to find the dnn implementation."
# After further research looks like it may be Tensorserve "locking" the folder / files
# and if you train your model at the same time with the same folers it gets that error.

# New Links
# https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model?version=nightly
# https://www.tensorflow.org/api_docs/python/tf/keras/models/save_model?version=nightly
# https://www.tensorflow.org/guide/keras/save_and_serialize
# https://tutorialdeep.com/knowhow/round-float-to-2-decimal-places-python/
# Getting started with Tensorflow in Google Colaboratory https://www.youtube.com/watch?v=PitcORQSjNM
# Get started with Google Colab https://www.youtube.com/watch?v=inN8seMm7UI
# https://jupyter.org/install
# https://research.google.com/colaboratory/local-runtimes.html
# https://www.analyticsvidhya.com/blog/2016/08/evolution-core-concepts-deep-learning-neural-networks/

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
# https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/