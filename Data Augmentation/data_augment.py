# importing data
import tensorflow_datasets as tfds
pcam, pcam_info = tfds.load("patch_camelyon", data_dir = "./tensorflow_datasets/", with_info=True, download = False)

import numpy as np
from random import seed
from random import sample
import cv2 as cv
import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from model import create_cnn
import preprocessing_techniques as helper


print("performing data augmentation!")

# Import TensorFlow and relevant Keras classes to setup the model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

print('--------------------- executing data_augment.py --------------------------')

# define CNN architecture
model = create_cnn()

### Data Preprocessing

print("preparing training material")
pcam_train = list(pcam['train'])
pcam_train_length = len(pcam_train)

PERCENTAGE_AUGMENT = 1.2

seed(42)
pcam_train_base = sample(pcam_train, int(PERCENTAGE_AUGMENT/2* pcam_train_length))
seed(121)
pcam_train_altered = sample(pcam_train, int(PERCENTAGE_AUGMENT/2*pcam_train_length))

pcam_train_img = [helper.convert_base_img(pcam_train_base[i]['image']) for i in range(len(pcam_train_base))]
pcam_train_label = [helper.convert_base_label(pcam_train_base[i]['label']) for i in range(len(pcam_train_base))]

helper.generate_contrast_image([pcam_train_altered[i]['image'] for i in range(len(pcam_train_altered))], pcam_train_img)
helper.generate_labels([pcam_train_altered[i]['label'] for i in range(len(pcam_train_altered))], pcam_train_label)
print(f"training length is {len(pcam_train_base) + len(pcam_train_altered)}")
train_transformed = tf.data.Dataset.from_tensor_slices((pcam_train_img, pcam_train_label))
train_pipeline = train_transformed.shuffle(1024).repeat().batch(64).prefetch(2)


print('preparing validation material')
pcam_valid = list(pcam['validation'])
pcam_valid_length = len(pcam_valid)

seed(1)
pcam_valid_base = sample(pcam_valid, int(PERCENTAGE_AUGMENT/2 * pcam_valid_length))
seed(131)
pcam_valid_altered = sample(pcam_valid, int(PERCENTAGE_AUGMENT/2 * pcam_valid_length))

pcam_valid_img = [helper.convert_base_img(pcam_valid_base[i]['image']) for i in range(len(pcam_valid_base))]
pcam_valid_label = [helper.convert_base_label(pcam_valid_base[i]['label']) for i in range(len(pcam_valid_base))]

helper.generate_contrast_image([pcam_valid_altered[i]['image'] for i in range(len(pcam_valid_altered))], pcam_valid_img)
helper.generate_labels([pcam_valid_altered[i]['label'] for i in range(len(pcam_valid_altered))], pcam_valid_label)
print(f"validation length is {len(pcam_valid_base) + len(pcam_valid_altered)}")
valid_transformed = tf.data.Dataset.from_tensor_slices((pcam_valid_img, pcam_valid_label))
validation_pipeline = valid_transformed.repeat().batch(128).prefetch(2)

# train model
print("begining training...")
hist = model.fit(train_pipeline, validation_data=validation_pipeline, verbose=2, epochs=10, steps_per_epoch=2048, validation_steps=256)


# Test material
print("generate test material...")

# test 1: baseline data
test_pipeline_base = pcam['test'].map(helper.convert_sample, num_parallel_calls=8).batch(128).prefetch(2)
print("Test set accuracy for original images is {0:.4f}".format(model.evaluate(test_pipeline_base, steps=128, verbose=0)[1]))

# test 2: transformed data

list_test = list(pcam['test'])
test_img = [list_test[i]['image'] for i in range(len(list_test))]
test_label = [list_test[i]['label'] for i in range(len(list_test))]
test_label_convert = []
helper.generate_labels(test_label, test_label_convert)
test_image_convert = []
helper.generate_contrast_image(test_img, test_image_convert)
test_transformed = tf.data.Dataset.from_tensor_slices((test_image_convert, test_label_convert))
test_pipeline = test_transformed.batch(128).prefetch(2)
print("Test set accuracy for contrast images is is {0:.4f}".format(model.evaluate(test_pipeline, steps=128, verbose=0)[1]))








