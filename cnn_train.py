# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 08:35:25 2019

@author: vibhor
"""

from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
BATCH_SIZE = 50
EPOCH_COUNT = 128
MODEL_PATH = 'createdmodel.hdf5'
WEIGHTS_PATH = 'weights1.hdf5'
depth = 3
width = 192
height = 192
shape= (depth, height, width)
if K.image_data_format() == 'channels_last':
    shape = (height, width, depth)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = shape))
model.add(ZeroPadding2D(padding = (1, 1)))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))
model.add(Dropout(rate = 0.25))
model.add(ZeroPadding2D(padding = (1, 1)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(ZeroPadding2D(padding = (1, 1)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))
model.add(Dropout(rate = 0.25))
model.add(ZeroPadding2D(padding = (1, 1)))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
model.add(ZeroPadding2D(padding = (1, 1)))
model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides=(2, 2)))
model.add(Dropout(rate = 0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(
        'dataset/train',
        target_size = (height, width),
        batch_size = BATCH_SIZE,
        class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(
        'dataset/test',
        target_size = (height, width),
        batch_size = BATCH_SIZE,
        class_mode = 'categorical')
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit_generator(
        training_set,
        epochs = EPOCH_COUNT,
        steps_per_epoch=BATCH_SIZE,validation_steps=64,
        validation_data = test_set,
        callbacks = [earlystop, checkpoint])
model.save(MODEL_PATH, True, True)