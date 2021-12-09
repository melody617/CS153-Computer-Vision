import cv2 as cv
import tensorflow as tf

# preprocess images
def generate_contrast_image(input_image_list, output_image_list):
    for image in input_image_list:

        # source: https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
        # convert to HSV
        img_hsv = cv.cvtColor(image.numpy(), cv.COLOR_RGB2HSV)
        img_hsv[:, :, 2] = cv.equalizeHist(img_hsv[:, :, 2])
        contrast_output = cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)
        output_image_list+= [tf.convert_to_tensor(contrast_output, dtype=tf.float32)]

# convert original image into format that can be taken in by TF
def convert_base_img(image):
    return tf.image.convert_image_dtype(image, tf.float32)

def generate_blur_image(input_image_list, output_image_list):
    for image in input_image_list:
        blurred = cv.bilateralFilter(image.numpy(),9,80,80)
        output_image_list+= [tf.convert_to_tensor(blurred, dtype=tf.float32)]

# preprocess labels
def generate_labels(input_label_list, output_label_list):
    for label in input_label_list:
        try:
            output_label_list.append(tf.one_hot(int(label), 2, dtype=tf.float32))
        except TypeError:
            print("type error in generating labels.")
            output_label_list.append(0)

def convert_base_label(label):
    return tf.one_hot(label, 2, dtype=tf.float32)

def convert_sample(sample):
    image, label = sample['image'], sample['label']  
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.one_hot(label, 2, dtype=tf.float32)
    return image, label