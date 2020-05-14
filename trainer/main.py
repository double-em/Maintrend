import numpy as np
import pandas as pd
import os
import sys
import datetime
import requests
import json
import logging
import importlib
from pathlib import Path
import util.data_puller as api
import util.difference_holder as dh

trainer_logger = logging.getLogger('trainer')
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(name)s: %(message)s")
trainer_logger.setLevel(logging.DEBUG)

base_url = os.environ['API_BASE_URL'] + '/' + os.environ['API_CHANNEL'] + '/' + os.environ['API_F']
apikey = os.environ['API_KEY']

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



model_name = "test_datapull"
log_dir = "logs"
model_dir = "models"

_patience = 60

_batch_size = 16
_buffer_size = 10000

_max_epochs = 1000
_back_in_time = 60 # Days
_step = 1 # Days to offset next dataset
_target_size = 1 # How many to predict

build_mode = True
test_mode = True
save_mode = True



### Hyperparamters
hp_hidden_num_layers = hp.HParam('hidden_num_layers', hp.IntInterval(0, 4))
hp_optimizer = hp.HParam('optimizer', hp.Discrete(['nadam', 'adam', 'rmsprop', 'sgd']))
hp_output_units = hp.HParam('output_units', hp.Discrete([50, 300, 600]))

hp.hparams_config(
    hparams=[hp_hidden_num_layers, hp_optimizer, hp_output_units],
    metrics=[hp.Metric('mae', display_name="Mean Absolute Error")]
)

if build_mode:
    hp_hidden_num_layers = hp.HParam('hidden_num_layers', hp.IntInterval(3, 3))
    hp_optimizer = hp.HParam('optimizer', hp.Discrete(['rmsprop']))
    hp_output_units = hp.HParam('output_units', hp.Discrete([300]))



### Optimizers
_optimizer = keras.optimizers.Nadam()
# keras.optimizers.RMSprop()
# keras.optimizers.Nadam()
# keras.optimizers.Adam()

### Losses
_loss = keras.losses.mean_absolute_error
# _loss = keras.losses.mean_squared_error

print()
trainer_logger.debug(f"Visible Devices: {tf.config.get_visible_devices()}")

# train_old = api.pulldata2()
train = api.apicallv3(_back_in_time, base_url, apikey, "2099-01-31 00:00:00", "2000-01-31 00:00:00")
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
def get_callbacks(name, hparams, log_path):
    return [
        EarlyStopping(monitor="val_loss", patience=_patience, restore_best_weights=True),
        TensorBoard(
            log_dir=log_path,
            profile_batch=2,
            histogram_freq=1
        ),
        hp.KerasCallback(log_path, hparams, name)
    ]



### Compile and Fit
def compile_and_fit(model, name, hparams, version, optimizer=_optimizer, loss=_loss, max_epochs=_max_epochs):
    
    log_dir_path = log_dir + "/" + name + "/" + str(version)
    trainer_logger.debug(f"Model log directory: {log_dir_path}")

    trainer_logger.info(f"Compiling model {name}...")
    model.compile(loss=loss, optimizer=optimizer)

    trainer_logger.info(f"Fitting model {name}...")

    model_history = model.fit(
        train_dataset, 
        epochs=max_epochs, 
        validation_data=val_dataset, 
        verbose=1, 
        callbacks=get_callbacks(name, hparams, log_dir_path))

    print()
    
    return model_history



### Dynamic model builder
def model_builder(name, hparams):

    trainer_logger.info(f"Building model {name}...")

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



### Test model
def test_model(model):
    
    treshold = 3
    differ = dh.DifferenceHolder(treshold, trainer_logger, True)

    trainer_logger.debug(f"Testing model with {treshold} day(s) treshold...")
    predictions = model.predict(test_dataset)

    # NOTE: Shape = (6, 2, 16, 60, 4)
    # NOTE: Shape = (batches, (x and y), batch_size, history_size, parameters)
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
    
    differ.PrintFinal()



def save_model(model, path):
    # NOTE: Model version is NEEDED or Tensorflow Serve cant find any "serverable versions"
    trainer_logger.info(f"Saving model '{model_name}' at: {path}")
    model.save(path)



### Trainer loop
session_version = 1

for output_units in hp_output_units.domain.values:
    for hidden_num_layers in range(hp_hidden_num_layers.domain.min_value, (hp_hidden_num_layers.domain.max_value + 1)):
        for optimizer in hp_optimizer.domain.values:
            hparams = {
                hp_hidden_num_layers: hidden_num_layers,
                hp_optimizer: optimizer,
                hp_output_units: output_units
            }
            
            trainer_logger.info(f"Starting session: {session_version}")
            trainer_logger.debug(f"Using Hyperparameters: {hparams[h] for h in hparams}")

            model_tmp = model_builder(model_name, hparams)
            compile_and_fit(model_tmp, model_tmp.name, hparams, session_version, hparams[hp_optimizer])

            if test_mode:
                test_model(model_tmp)

            if save_mode:
                model_save_path = model_dir + "/" + model_tmp.name + "/" + str(session_version)
                save_model(model_tmp, model_save_path)

            session_version += 1



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