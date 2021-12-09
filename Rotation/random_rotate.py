import tensorflow_datasets as tfds
pcam, pcam_info = tfds.load("patch_camelyon", data_dir = "./tensorflow_datasets/", with_info=True, download = False)
import numpy as np

import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import preprocessing_techniques as helper

print("random_rotate...")

# Import TensorFlow and relevant Keras classes to setup the model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

input_img = Input(shape=(96,96,3))

# Now we define the layers of the convolutional network: three blocks of two convolutional layers and a max-pool layer.
x = data_augmentation(input_img)
x = Conv2D(16, (3, 3), padding='valid', activation='relu')(x)
x = Conv2D(16, (3, 3), padding='valid', activation='relu')(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
x = Conv2D(32, (3, 3), padding='valid', activation='relu')(x)
x = Conv2D(32, (3, 3), padding='valid', activation='relu')(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

# Now we flatten the output from a 4D to a 2D tensor to be able to use fully-connected (dense) layers for the final
# classification part. Here we also use a bit of dropout for regularization. The last layer uses a softmax to obtain class
# likelihoods (i.e. metastasis vs. non-metastasis)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(rate=0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(rate=0.2)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=input_img, outputs=predictions)
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

model.save("./patchcamelyon10epoch.hf5")










''' testing:
test_photo = list(pcam['test'])[0]['image']


augmented_image = data_augmentation(test_photo)
for i in range(9):
  augmented_image = data_augmentation(test_photo)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image)
  plt.axis("off")
plt.show()
'''
