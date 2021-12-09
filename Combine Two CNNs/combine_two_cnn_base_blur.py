print("combine two cnn original + blur")

from model import create_cnn
import tensorflow_datasets as tfds
pcam, pcam_info = tfds.load("patch_camelyon", data_dir = "./tensorflow_datasets/", with_info=True, download = False)
import numpy as np
import cv2 as cv
import tensorflow as tf
from model import create_cnn
import preprocessing_techniques as helper
import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# define CNN
model = create_cnn()


# gather train material

print("generate training material...")
list_train = list(pcam['train'])
list_img = [list_train[i]['image'] for i in range(len(list_train))]
list_label = [list_train[i]['label'] for i in range(len(list_train))]

# generate labels
list_label_convert = []
helper.generate_labels(list_label, list_label_convert)

list_image_convert = []
helper.generate_blur_image(list_img, list_image_convert)
list_image_baseline = [helper.convert_base_img(list_img[i]) for i in range(len(list_img))]

print("generate validation material...")
list_valid = list(pcam['validation'])
valid_img = [list_valid[i]['image'] for i in range(len(list_valid))]
valid_label = [list_valid[i]['label'] for i in range(len(list_valid))]
valid_label_convert = []
helper.generate_labels(valid_label, valid_label_convert)
valid_image_convert = []
helper.generate_blur_image(valid_img, valid_image_convert)
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
helper.generate_blur_image(test_img, test_image_convert)
test_image_baseline = [helper.convert_base_img(test_img[i]) for i in range(len(list_valid))]

print("Test set accuracy is {0:.4f}".format(model.evaluate(x = [np.asarray(test_image_baseline), np.asarray(test_image_convert)], y = np.asarray(test_label_convert), batch_size = 64, steps = 128, verbose= 0)[1]))
