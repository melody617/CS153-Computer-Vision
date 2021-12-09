import tensorflow_datasets as tfds
pcam, pcam_info = tfds.load("patch_camelyon", data_dir = "./tensorflow_datasets/", with_info=True, download = False)
import numpy as np

import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from model import create_cnn
import preprocessing_techniques as helper

print("BASELINE MODEL")
model = create_cnn()

# create train and validation pipeline
train_pipeline = pcam['train'].map(helper.convert_sample,
                                   num_parallel_calls=8).shuffle(1024).repeat().batch(64).prefetch(2)
valid_pipeline = pcam['validation'].map(helper.convert_sample,
                                        num_parallel_calls=8).repeat().batch(128).prefetch(2)


hist = model.fit(train_pipeline, validation_data=valid_pipeline, verbose=2, epochs=10, steps_per_epoch=2048, validation_steps=256)

test_pipeline = pcam['test'].map(helper.convert_sample, num_parallel_calls=8).batch(128).prefetch(2)
print("Test set accuracy is {0:.4f}".format(model.evaluate(test_pipeline, steps=128, verbose=0)[1]))

model.save("./patchcamelyon10epoch.hf5")
