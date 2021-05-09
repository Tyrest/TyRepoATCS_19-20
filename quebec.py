import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Conv2D, MaxPooling2D

from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Nadam

from keras import regularizers
from keras.layers import Dropout, SpatialDropout2D
from keras.callbacks import LearningRateScheduler, EarlyStopping

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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# data augumetation
datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        )

# data generator model to train and validation set
train_gen = datagen.flow(x_train, y_train)

print("Original input shape: " + str(x_train.shape[1:]))

def lr_schedule(epoch):
	lr = 1e-3
	if epoch > 196:
		lr *= 1e-3
	elif epoch > 128:
		lr *= 1e-2
	elif epoch > 64:
		lr *= 1e-1
	return lr

def test_model(model):

	# learning schedule callback
	lr_scheduler = LearningRateScheduler(lr_schedule)

	early_stopper = EarlyStopping(monitor='val_accuracy',
                                 min_delta=0.00001,
                                 patience=64,
                                 restore_best_weights=True)

	t0 = time.time()
	history = model.fit(x_train, y_train,
						epochs=epochs,
						verbose=2,
						validation_data=(x_test, y_test),
						use_multiprocessing=True,
						shuffle=True,
						callbacks=[lr_scheduler, early_stopper])
	print("Took " + str(time.time() - t0) + " seconds to fit")
	score = model.evaluate(x_test, y_test, verbose=0)
	print('\nTest accuracy:', score[1])

	model.save("model.h5")

def test_model_img_pre(model):

	# learning schedule callback
	lr_scheduler = LearningRateScheduler(lr_schedule)

	early_stopper = EarlyStopping(monitor='val_accuracy',
                                 min_delta=0.00001,
                                 patience=64,
                                 restore_best_weights=True)

	batch_size = 32

	t0 = time.time()
	history = model.fit(train_gen,
						epochs=epochs,
						verbose=2,
						validation_data=(x_test, y_test),
						shuffle=True,
						use_multiprocessing=True,
						callbacks=[lr_scheduler, early_stopper])
	print("Took " + str(time.time() - t0) + " seconds to fit")
	score = model.evaluate(x_test, y_test, verbose=0)
	print('\nTest accuracy:', score[1])

	model.save("model.h5")

# ~61% accuracy without CNN
def model1():
	layers = [512, 512, 512, 512, 256]
	act = 'relu'
	init = 'he_uniform'
	mloss = 'categorical_crossentropy'
	opt = RMSprop()

	model = Sequential()
	model.add(Flatten(input_shape=x_train.shape[1:]))

	drop = 0.2

	for layer in layers:
		model.add(Dense(layer, kernel_initializer=init, use_bias=False))
		model.add(Activation(act))
		model.add(BatchNormalization())
		model.add(Dropout(drop))
		if drop < 0.5:
			drop += 0.1

	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=mloss,
	              optimizer=opt,
	              metrics=['accuracy'])

	return model

# CNN ~92.85%
def model2():
	init = 'he_uniform'
	opt = RMSprop()
	mloss = 'categorical_crossentropy'
	act = 'elu'

	model = Sequential()

	model.add(Conv2D(256, (3, 3), padding='same', input_shape=x_train.shape[1:], kernel_initializer=init))
	model.add(Activation(act))
	model.add(BatchNormalization())
	model.add(Conv2D(256, (3, 3), kernel_initializer=init))
	model.add(Activation(act))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(SpatialDropout2D(0.3))

	model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=init))
	model.add(Activation(act))
	model.add(BatchNormalization())
	model.add(Conv2D(512, (3, 3), kernel_initializer=init))
	model.add(Activation(act))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(SpatialDropout2D(0.4))

	model.add(Conv2D(1024, (3, 3), padding='same', kernel_initializer=init))
	model.add(Activation(act))
	model.add(BatchNormalization())
	model.add(Conv2D(1024, (3, 3), kernel_initializer=init))
	model.add(Activation(act))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(SpatialDropout2D(0.5))

	model.add(Flatten())

	model.add(Dense(512, kernel_initializer=init))
	model.add(Activation(act))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(512, kernel_initializer=init))
	model.add(Activation(act))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	# Compile the model
	model.compile(optimizer=opt, loss=mloss, metrics=['accuracy'])

	return model

epochs = 512

# current_model = model1()
current_model = model2()

current_model.summary()

# test_model(current_model)
test_model_img_pre(current_model)
