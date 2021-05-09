import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# Which model to evaluate
model_to_test = '1-21-1100-9083.h5'

# CIFAR10
# 50000/10000 32x32 color images of various real-world items (cars, ships, etc...)
# https://www.cs.toronto.edu/~kriz/cifar.html
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preparing the data:
print("Scaling input data...")
max_val = np.max(x_train).astype(np.float32)
print("Max value: " +  str(max_val))
x_train = x_train.astype(np.float32) / max_val
x_test = x_test.astype(np.float32) / max_val
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# Convert class vectors to binary class matrices.
num_classes = len(np.unique(y_train))
print("Number of classes in this dataset: " + str(num_classes))
if num_classes > 2:
	print("One hot encoding targets...")
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

model = load_model(model_to_test)
model.summary()

print("Testing model...")
score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest accuracy:', score[1])
