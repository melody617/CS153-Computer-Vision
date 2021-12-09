from model import create_cnn
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from model import create_cnn
import preprocessing_techniques as helper

print("Histgoram Equalization")
print("train on histogram equalizationi, test on original")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


import tensorflow_datasets as tfds
pcam, pcam_info = tfds.load("patch_camelyon", data_dir = "./tensorflow_datasets/", with_info=True, download = False)
print(pcam_info)

# define CNN
model = create_cnn()

# Training Material
print("generate training material...")
list_train = list(pcam['train'])
list_img = [list_train[i]['image'] for i in range(len(list_train))]
list_label = [list_train[i]['label'] for i in range(len(list_train))]
list_label_convert = []
helper.generate_labels(list_label, list_label_convert)
list_image_convert = []
helper.generate_contrast_image(list_img, list_image_convert)
train_transformed = tf.data.Dataset.from_tensor_slices((list_image_convert, list_label_convert))
train_pipeline = train_transformed.shuffle(1024).repeat().batch(64).prefetch(2)

# Validation Material
print("generate validation material...")
valid_pipeline = pcam['validation'].map(helper.convert_sample,
                                        num_parallel_calls=8).repeat().batch(128).prefetch(2)

# Training
print("begining training...")
hist = model.fit(train_pipeline, validation_data=valid_pipeline, verbose=2, epochs=10, steps_per_epoch=2048, validation_steps=256)

# Test material
print("generate test material...")
test_pipeline = pcam['test'].map(helper.convert_sample, num_parallel_calls=8).batch(128).prefetch(2)
print("Test set accuracy is {0:.4f}".format(model.evaluate(test_pipeline, steps=128, verbose=0)[1]))
