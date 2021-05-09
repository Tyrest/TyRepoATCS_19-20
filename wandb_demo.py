# Helper libraries
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Import and initialize wandb
# You must log into wandb in your terminal
import wandb
from wandb.keras import WandbCallback
wandb.init(project="demo")

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

print("Original input shape: " + str(x_train.shape[1:]))

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, BatchNormalization

from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam

from keras import regularizers
from keras.layers import Dropout
from keras.layers import AlphaDropout

hidden1 = 100
act = 'relu'
init = 'he_uniform'
mloss = 'categorical_crossentropy'
opt = Adagrad()

model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(hidden1, kernel_initializer=init, use_bias=False))
model.add(BatchNormalization())
model.add(Activation(act))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=mloss,
              optimizer=opt,
              metrics=['accuracy'])

epochs = 20

print("\nTraining with " +  act + ", " + init + " and BatchNormalization:")
# Add WandbCallback() to callbacks
history = model.fit(x_train, y_train,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test),
              		shuffle=True,
                    callbacks=[WandbCallback()],
                    use_multiprocessing=True)

score = model.evaluate(x_test, y_test, verbose=0)
print('\nTest accuracy:', score[1])

# Save results to wandb directory
model.save(os.path.join(wandb.run.dir, "model.h5"))
