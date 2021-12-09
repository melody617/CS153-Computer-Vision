
import tensorflow_datasets as tfds
pcam, pcam_info = tfds.load("patch_camelyon", data_dir = "./tensorflow_datasets/", with_info=True, download = False)
import numpy as np
import cv2 as cv
import preprocessing_techniques as helper

import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

print("combine two cnn baseline + rotate!")

# Import TensorFlow and relevant Keras classes to setup the model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers.merge import concatenate
from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])



# CNN for input 1
input_img1 = Input(shape=(96,96,3))
x = Conv2D(16, (3, 3), padding='valid', activation='relu')(input_img1)
x = Conv2D(16, (3, 3), padding='valid', activation='relu')(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
x = Conv2D(32, (3, 3), padding='valid', activation='relu')(x)
x = Conv2D(32, (3, 3), padding='valid', activation='relu')(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(rate=0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(rate=0.2)(x)

# CNN for input 2
x2 = data_augmentation(input_img1) # apply random rotation and horizontal + vertical flipping
x2 = Conv2D(16, (3, 3), padding='valid', activation='relu')(x2)
x2 = Conv2D(16, (3, 3), padding='valid', activation='relu')(x2)
x2 = MaxPool2D(pool_size=(2,2), strides=(2,2))(x2)
x2 = Conv2D(32, (3, 3), padding='valid', activation='relu')(x2)
x2 = Conv2D(32, (3, 3), padding='valid', activation='relu')(x2)
x2 = MaxPool2D(pool_size=(2,2), strides=(2,2))(x2)
x2 = Conv2D(64, (3, 3), padding='valid', activation='relu')(x2)
x2 = Conv2D(64, (3, 3), padding='valid', activation='relu')(x2)
x2 = MaxPool2D(pool_size=(2,2), strides=(2,2))(x2)
x2 = Flatten()(x2)
x2 = Dense(256, activation='relu')(x2)
x2 = Dropout(rate=0.2)(x2)
x2 = Dense(128, activation='relu')(x2)
x2 = Dropout(rate=0.2)(x2)

output = concatenate([x, x2]) # outputs are concadenated togehter before passing to softmax
predictions = Dense(2, activation='softmax')(output)

# Now we define the inputs/outputs of the model and setup the optimizer. In this case we use regular stochastic gradient
# descent with Nesterov momentum. The loss we use is cross-entropy and we would like to output accuracy as an additional metric.
model = Model(inputs=input_img1, outputs=predictions)
sgd_opt = SGD(learning_rate=0.01, momentum=0.9, decay=0.0, clipnorm=1.0, nesterov=True)
model.compile(optimizer=sgd_opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# define train and validation pipeline
train_pipeline = pcam['train'].map(helper.convert_sample,
                                   num_parallel_calls=8).shuffle(1024).repeat().batch(64).prefetch(2)
valid_pipeline = pcam['validation'].map(helper.convert_sample,
                                        num_parallel_calls=8).repeat().batch(128).prefetch(2)


hist = model.fit(train_pipeline, validation_data=valid_pipeline, verbose=2, epochs=10, steps_per_epoch=2048, validation_steps=256)

test_pipeline = pcam['test'].map(helper.convert_sample, num_parallel_calls=8).batch(128).prefetch(2)
print("Test set accuracy is {0:.4f}".format(model.evaluate(test_pipeline, steps=128, verbose=0)[1]))