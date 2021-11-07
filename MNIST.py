# https://harishnarayanan.org/writing/artistic-style-transfer/

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.cmap'] = 'Greys'

import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

print(train_images.shape)
print(test_images.shape)