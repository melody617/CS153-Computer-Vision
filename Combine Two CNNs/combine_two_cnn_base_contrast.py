print("combine two cnn full! + contrast")

import tensorflow_datasets as tfds
pcam, pcam_info = tfds.load("patch_camelyon", data_dir = "./tensorflow_datasets/", with_info=True, download = False)
import numpy as np
import cv2 as cv
import preprocessing_techniques as helper

import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)



# Import TensorFlow and relevant Keras classes to setup the model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers.merge import concatenate

input_img1 = Input(shape=(96,96,3))

# Now we define the layers of the convolutional network: three blocks of two convolutional layers and a max-pool layer.
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

# input 2
input_img2 = Input(shape=(96,96,3))
x2 = Conv2D(16, (3, 3), padding='valid', activation='relu')(input_img2)
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

output = concatenate([x, x2])
predictions = Dense(2, activation='softmax')(output)

# define optimizer and learning rates
model = Model(inputs=[input_img1, input_img2], outputs=predictions)
sgd_opt = SGD(learning_rate=0.01, momentum=0.9, decay=0.0, clipnorm=1.0, nesterov=True)
model.compile(optimizer=sgd_opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# define training material

print("generate training material...")
list_train = list(pcam['train'])
list_img = [list_train[i]['image'] for i in range(len(list_train))]
list_label = [list_train[i]['label'] for i in range(len(list_train))]

# generate labels
list_label_convert = []
helper.generate_labels(list_label, list_label_convert)

list_image_convert = []
helper.generate_contrast_image(list_img, list_image_convert)
list_image_baseline = [helper.convert_base_img(list_img[i]) for i in range(len(list_img))]

print("generate validation material...")
list_valid = list(pcam['validation'])
valid_img = [list_valid[i]['image'] for i in range(len(list_valid))]
valid_label = [list_valid[i]['label'] for i in range(len(list_valid))]
valid_label_convert = []
helper.generate_labels(valid_label, valid_label_convert)
valid_image_convert = []
helper.generate_contrast_image(valid_img, valid_image_convert)
valid_image_baseline = [helper.convert_base_img(valid_img[i]) for i in range(len(list_valid))]

hist =model.fit([np.asarray(list_image_baseline), np.asarray(list_image_convert)], np.array(list_label_convert), epochs = 10, validation_data=([np.asarray(valid_image_baseline), np.asarray(valid_image_convert)], np.asarray(valid_label_convert)), steps_per_epoch	= 2048, verbose = 2, validation_steps = 256)


# Test material
print("generate test material...")
list_test = list(pcam['test'])
test_img = [list_test[i]['image'] for i in range(len(list_test))]
test_label = [list_test[i]['label'] for i in range(len(list_test))]
test_label_convert = []
helper.generate_labels(test_label, test_label_convert)
test_image_convert = []
helper.generate_contrast_image(test_img, test_image_convert)
test_image_baseline = [helper.convert_base_img(test_img[i]) for i in range(len(list_valid))]

print("Test set accuracy is {0:.4f}".format(model.evaluate(x = [np.asarray(test_image_baseline), np.asarray(test_image_convert)], y = np.asarray(test_label_convert), batch_size = 64, steps = 128, verbose= 0)[1]))
