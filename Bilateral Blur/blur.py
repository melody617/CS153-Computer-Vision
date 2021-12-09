import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

print('BLUR')
print("trained on blur images, tested on blurred images")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from model import create_cnn
import preprocessing_techniques as helper
import tensorflow as tf

import tensorflow_datasets as tfds
pcam, pcam_info = tfds.load("patch_camelyon", data_dir = "./tensorflow_datasets/", with_info=True, download = False)
print(pcam_info)

# define the CNN architecture used
model = create_cnn()


# Training Material
print("generate training material...")
list_train = list(pcam['train'])
list_img = [list_train[i]['image'] for i in range(len(list_train))]
list_label = [list_train[i]['label'] for i in range(len(list_train))]
list_label_convert = []
helper.generate_labels(list_label, list_label_convert)
list_image_convert = []
helper.generate_blur_image(list_img, list_image_convert)
train_transformed = tf.data.Dataset.from_tensor_slices((list_image_convert, list_label_convert))
train_pipeline = train_transformed.shuffle(1024).repeat().batch(64).prefetch(2)

# Validation Material
print("generate validation material...")
list_valid = list(pcam['validation'])
valid_img = [list_valid[i]['image'] for i in range(len(list_valid))]
valid_label = [list_valid[i]['label'] for i in range(len(list_valid))]
valid_label_convert = []
helper.generate_labels(valid_label, valid_label_convert)
valid_image_convert = []
helper.generate_blur_image(valid_img, valid_image_convert)
valid_transformed = tf.data.Dataset.from_tensor_slices((valid_image_convert, valid_label_convert))
validation_pipeline = valid_transformed.repeat().batch(128).prefetch(2)

# Training
print("begining training...")
hist = model.fit(train_pipeline, validation_data=validation_pipeline, verbose=2, epochs=10, steps_per_epoch=2048, validation_steps=256)

# Test material
print("generate test material...")
list_test = list(pcam['test'])
test_img = [list_test[i]['image'] for i in range(len(list_test))]
test_label = [list_test[i]['label'] for i in range(len(list_test))]
test_label_convert = []
helper.generate_labels(test_label, test_label_convert)
test_image_convert = []
helper.generate_blur_image(test_img, test_image_convert)
test_transformed = tf.data.Dataset.from_tensor_slices((test_image_convert, test_label_convert))
test_pipeline = test_transformed.batch(128).prefetch(2)
print("Test set accuracy is {0:.4f}".format(model.evaluate(test_pipeline, steps=128, verbose=0)[1]))

model.save("./patchcamelyonblur.hf5")