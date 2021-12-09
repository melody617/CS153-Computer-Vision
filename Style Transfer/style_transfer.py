import tensorflow_datasets as tfds
pcam, pcam_info = tfds.load("patch_camelyon", data_dir = "./tensorflow_datasets/", with_info=True, download = False)
print(pcam_info)
from PIL import Image
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import glob
import functools
from model import create_cnn
import tensorflow as tf

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

print('style transfer!')

# define CNN architecture
model = create_cnn()

# defining inputs
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

tf.executing_eagerly()
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
style_images = glob.glob("./style_transfer_imgs/*")
print(style_images)
print(f"style image is {style_images[3]}")
style_image = load_img(style_images[3])


def preprocess_content(img):
    max_dim = 512
#     img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    
    ######### newly added
    image = hub_model(tf.constant(img), tf.constant(style_image))[0]
    image = tensor_to_image(image)
    image = image.resize((96,96))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

print("preparing training images...")
list_train = list(pcam['train'])
list_img = [list_train[i]['image'] for i in range(len(list_train))]
list_label = [list_train[i]['label'] for i in range(len(list_train))]

list_img_convert = []
for i, img in enumerate(list_img):
    list_img_convert.append(preprocess_content(img))
list_label_convert = []
for i, label in enumerate(list_label):
    try:
        list_label_convert.append(tf.one_hot(int(label), 2, dtype=tf.float32))
    except TypeError:
        list_label_convert.append(list_label[0])

train_transformed = tf.data.Dataset.from_tensor_slices((list_img_convert, list_label_convert))
train_pipeline = train_transformed.shuffle(1024).repeat().batch(64).prefetch(2)

# Validate
print("preparing validation images...")

list_validation = list(pcam['validation'])
validation_img = [list_validation[i]['image'] for i in range(len(list_validation))]
validation_label = [list_validation[i]['label'] for i in range(len(list_validation))]
validation_img_convert = []
for i, img in enumerate(validation_img):
    validation_img_convert.append(preprocess_content(img))
validation_label_convert = []
for i, label in enumerate(validation_label):
    try:
        validation_label_convert.append(tf.one_hot(int(label), 2, dtype=tf.float32))
    except TypeError:
        validation_label_convert.append(list_label[0])

valid_transformed = tf.data.Dataset.from_tensor_slices((validation_img_convert, validation_label_convert))
valid_pipeline = valid_transformed.repeat().batch(128).prefetch(2)

print("preparing to train CNN...")
hist = model.fit(train_pipeline,
                 validation_data=valid_pipeline,
                 verbose=2, epochs=10, steps_per_epoch=2048, validation_steps=256)

print("preparing test images...")
list_test = list(pcam['test'])
test_img = [list_test[i]['image'] for i in range(len(list_test))]
test_label = [list_test[i]['label'] for i in range(len(list_test))]
test_img_convert = []
for i, img in enumerate(test_img):
    test_img_convert.append(preprocess_content(img))
test_label_convert = []
for i, label in enumerate(test_label):
    try:
        test_label_convert.append(tf.one_hot(int(label), 2, dtype=tf.float32))
    except TypeError:
        test_label_convert.append(list_label[0])

test_transformed = tf.data.Dataset.from_tensor_slices((test_img_convert, test_label_convert))
test_pipeline = test_transformed.shuffle(1024).batch(128).prefetch(2)
print("Test set accuracy is {0:.4f}".format(model.evaluate(test_pipeline, steps=128, verbose=0)[1]))
